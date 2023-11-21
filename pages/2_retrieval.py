import requests
import streamlit as st
import pickle
import pandas as pd
import base64
import numpy as np
import ujson as json
from io import StringIO
import re
from rank_bm25 import BM25Okapi
import random
import os
import gzip
import bz2
import csv
import ujson as json
import glob
import math
import torch
import torch.nn as nn


st.title('Table Retrieval')

class pipeline:
    def __init__(self, row_model, col_model, file_path, data_file_names, representation_file_names=None, BM25_k=300, top_k=5):
        self.row_model = row_model
        self.col_model = col_model
        self.data_file_names = []
        self.representation_file_names = []
        # self.file_path = file_path
        for file_name in data_file_names:
            self.data_file_names.append(file_path + file_name)
        if(representation_file_names is not None):
            for file_name in representation_file_names:
                self.representation_file_names.append(file_path + file_name)
        self.BM25_k = BM25_k
        self.top_k = top_k

    def read_file(self, in_file,binary=False,errors=None):
        if binary:
            if in_file.endswith('.gz'):
                return gzip.open(in_file,'rb')
            elif in_file.endswith('.bz2'):
                return bz2.open(in_file,'rb')
            else:
                return open(in_file,'rb')

        else:
            if in_file.endswith('.gz'):
                return gzip.open(in_file,'rt',encoding='utf-8',errors=errors)
            elif in_file.endswith('.bz2'):
                return bz2.open(in_file,'rt',encoding='utf-8',errors=errors)
            else:
                return open(in_file,'r',encoding='utf-8',errors=errors)

    def pre_process(self, path):
        di = {}
        punc_pattern = r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]"
        with self.read_file(path) as fp:
            for n1,line in enumerate(fp):
                data = json.loads(line)
                for k,v in data.items():
                    qid = k
                    header = v[0]
                    rows = v[1:]
                    # print(qid,header,rows)
                    header1 = []
                    for h in header:
                        res = re.sub(punc_pattern,' ',h)
                        res = re.sub("\s+",' ',res)
                        header1.extend(res.lower().split())

                    rows1 = []
                    for i in rows:
                        for j in i:
                            res = re.sub(punc_pattern,' ',j)
                            res = re.sub("\s+",' ',res)
                            rows1.extend(res.lower().split())

                    header1.extend(rows1)
                    # print(header1)
                    di[k] = header1
        return di

    def preprocess_query(self, query):
        punc_pattern = r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]"
        res = re.sub(punc_pattern,' ',query)
        res = re.sub("\s+",' ',res)
        tokenized_query = res.lower().split()
        return tokenized_query


    def ranking_docs(self, query,di):
        tokenized_query = self.preprocess_query(query)
        bm25 = BM25Okapi(di.values())
        scores = bm25.get_scores(tokenized_query)
        ranked_documents = dict(sorted(zip(di.keys(), scores), key=lambda x: x[1], reverse=True))
        return ranked_documents

    def BM25(self, query):
        top = self.BM25_k
        paths = self.data_file_names
        di1 = self.pre_process(paths[0])
        # di2 = self.pre_process(paths[1])
        # di3 = self.pre_process(paths[2])
        ranked_doc1 = self.ranking_docs(query,di1)
        # ranked_doc2 = self.ranking_docs(query,di2)
        # ranked_doc3 = self.ranking_docs(query,di3)
        # result = {**ranked_doc1,**ranked_doc2,**ranked_doc3}
        result = {**ranked_doc1}
        final_result = dict(list(sorted(result.items(), key=lambda x: x[1], reverse=True))[:top])

        tables = {}
        for i in paths:
            with self.read_file(i) as fp:
                for n1,line in enumerate(fp):
                    data = json.loads(line)
                    for k,v in data.items():
                        if(k in final_result.keys()):
                            tables[k] = v

        return tables

    def getColRepresentation(self, header, rows):
        colRepresentation = []
        cols = [[str(h)] for h in header]
        for row in rows:
            for ci, cell in enumerate(row):
                if cell:  # for sparse table use case
                    cols[ci].append(str(cell))
        for col in cols:
            col_rep = ' * '.join(col)
            colRepresentation.append(col_rep)
        return colRepresentation

    def getRowRepresentation(self, header, rows):
        rowRepresentation = []
        for row in rows:
            row_rep = ' * '.join([h + ' : ' + str(c) for h, c in zip(header, row) if c])  # for sparse table use case
            rowRepresentation.append(row_rep)
        return rowRepresentation

    def get_data_from_table(self, header, rows, query):
        data = {'row_input':[], 'col_input':[], 'row_queries': [], 'col_queries': []}
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        max_seq_length = 128
        batch_size = 16
        col_queries = [query]*len(header)
        row_queries = [query]*len(rows)
        colsRepresentation = self.getColRepresentation(header, rows)
        rowsRepresentation = self.getRowRepresentation(header, rows)
        data['col_input'].extend(colsRepresentation)
        data['row_input'].extend(rowsRepresentation)
        data['col_queries'].extend(col_queries)
        data['row_queries'].extend(row_queries)
        col_encoding = tokenizer(data['col_input'], data['col_queries'],return_tensors='pt', padding='max_length', truncation='only_first', max_length=max_seq_length)
        row_encoding = tokenizer(data['row_input'], data['row_queries'],return_tensors='pt', padding='max_length', truncation='only_first', max_length=max_seq_length)
        return row_encoding, col_encoding

    def getLogits(self, header, rows, query):
        criterion = nn.CrossEntropyLoss()
        batch_size = 16
        row_data, col_data = self.get_data_from_table(header, rows, query)
        row_labels = torch.zeros([len(row_data),2])
        col_labels = torch.zeros([len(col_data), 2])
        rowsLogits = self.row_model(**row_data)[0].detach().cpu().numpy()[:, 1]
        colsLogits = self.col_model(**col_data)[0].detach().cpu().numpy()[:, 1]
        # _, _, rowsLogits = test(self.row_model, row_data, criterion, batch_size, row_labels)
        # _, _, colsLogits = test(self.col_model, col_data, criterion, batch_size, col_labels)
        return rowsLogits, colsLogits

    def getScores(self, rowsLogits, colsLogits):
        scores = []
        for i in range(len(rowsLogits)):
            for j in range(len(colsLogits)):
                score = float(rowsLogits[i] + colsLogits[j])
                scores.append([i,j,score])
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[0:self.top_k]

    def RCI(self, query, header, rows):
        batch_size = 16
        max_seq_length=128
        rowsLogits, colsLogits = self.getLogits(header, rows, query)

        # top_k = 5
        rci_scores = self.getScores(rowsLogits, colsLogits)
        return [{'row_ndx': i, 'col_ndx': j, 'confidence_score': score, 'text': rows[i][j]} for i, j, score in rci_scores]

    def RCI_System(self, query):
        tables = self.BM25(query)
        table_dict = {}
        table_scores = {}
        results = {}
        # print(tables)
        i=0
        for id, table in tables.items():
            if(i%10==0):
                print(i, " tables are scored")
            header = table[0]
            rows = table[1:]
            # print("header")
            # print(header)
            # print()
            # print('rows')
            # print(rows)
            table_dict[id] = {'header':header, 'rows': rows}
            cell_score = self.RCI(query, header, rows)
            results[id] = cell_score
            table_scores[id] = cell_score[0]['confidence_score']
            i+=1

        result_table_ids = sorted(table_scores, key=lambda k: table_scores[k], reverse=True)[:self.top_k]
        retrieved_results = []
        for id in result_table_ids:
            retrieved_results.append({'id':id, 'header':table_dict[id]['header'], 'rows':table_dict[id]['rows'], 'cells': results[id]})
        return retrieved_results


from transformers import (
    AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
    XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
    PreTrainedModel
)

# row_model_interaction = AlbertForSequenceClassification.from_pretrained('michaelrglass/albert-base-rci-wikisql-row')
# row_model_interaction.eval()
# torch.save(row_model_interaction.state_dict(), "./row_model_interaction")
# col_model_interaction = AlbertForSequenceClassification.from_pretrained('michaelrglass/albert-base-rci-wikisql-col')
# col_model_interaction.eval()
# torch.save(col_model_interaction.state_dict(), "./col_model_interaction")

row_model_interaction = AlbertForSequenceClassification.from_pretrained('michaelrglass/albert-base-rci-wikisql-row')
row_model_interaction.load_state_dict(torch.load("row_model_interaction"))
col_model_interaction = AlbertForSequenceClassification.from_pretrained('michaelrglass/albert-base-rci-wikisql-col')
col_model_interaction.load_state_dict(torch.load("col_model_interaction"))

pipe = pipeline(row_model_interaction, col_model_interaction, file_path="", data_file_names=['wiki_sql_data_lookup.jsonl'], BM25_k=10)



# query = "What Award was won in year 2008?"
query = st.text_input('Enter the query',"What was the award won in year 2006?")

if st.button("Click for Results"):
    st.write('The results are:')
    # st.write("Clicked", query)
# if query is not None:
    tables = pipe.RCI_System(query)
    for t in range(len(tables)):
        # print(tables[0]['cells'][0]['row_ndx'],tables[0]['cells'][0]['col_ndx'])
        # print(tables[0]['cells'][0]['row_ndx'],tables[0]['cells'][0]['col_ndx'])
        df = pd.DataFrame(tables[t]['rows'],columns=[tables[t]['header']])
        # st.write()
        # for t in tables:
        # df = pd.DataFrame(tables[0])
        st.dataframe(df.style.applymap(lambda _: "background-color: CornflowerBlue;", subset=(tables[t]['cells'][0]['row_ndx'],slice(None))))

# st.write(tables)



