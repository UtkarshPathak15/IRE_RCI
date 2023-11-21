import requests
import streamlit as st
import pickle
import pandas as pd
import base64
import numpy as np
import ujson as json
import glob
import math
import torch
import torch.nn as nn
from io import StringIO




st.set_page_config(page_title="Q/A")

st.title('Question/Answering')
# st.write("RCI")

from transformers import (
    AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
    XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
    PreTrainedModel
)



class pipeline:
    def __init__(self, row_model, col_model, top_k=5):
        self.row_model = row_model
        self.col_model = col_model
        self.top_k = top_k


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

    def RCI(self, query, table):
        batch_size = 16
        max_seq_length=128
        header = table['header']
        rows = table['rows']
        rowsLogits, colsLogits = self.getLogits(header, rows, query)

        # top_k = 5
        rci_scores = self.getScores(rowsLogits, colsLogits)
        return [{'row_ndx': i, 'col_ndx': j, 'confidence_score': score, 'text': rows[i][j]} for i, j, score in rci_scores]


row_model_interaction = AlbertForSequenceClassification.from_pretrained('michaelrglass/albert-base-rci-wikisql-row')
row_model_interaction.load_state_dict(torch.load("row_model_interaction"))
col_model_interaction = AlbertForSequenceClassification.from_pretrained('michaelrglass/albert-base-rci-wikisql-col')
col_model_interaction.load_state_dict(torch.load("col_model_interaction"))

pipe = pipeline(row_model_interaction, col_model_interaction)

def show_students(file):
    jsona = json.load(file)
    id=0
    header= []
    data = []
    for k,v in jsona.items():
        id = k
        header = v[0]
        data = v[1:]
    # df=pd.json_normalize(jsona)
    # st.write(header)
    table = {"id":id,"header":header,"rows":data}
    return table

# query = "In which hemisphere does the summer solstice occur in December?"
query = st.text_input('Enter the query',"In which hemisphere does the summer solstice occur in December?")

uploaded_file = st.file_uploader("Choose a file", type=['jsonl','json'])
if uploaded_file is not None:
    # st.write("inside")
    table = show_students(uploaded_file)
    # st.write(table)
    if st.button("Click for Results"):
        st.write('The results are:')

# table = {'header': ['Text1', 'HEMISPHERE', 'Text2', 'ORBITAL EVENT', 'Text3', 'MONTH OF OCCURENCE'],
#             'rows': [['In the', 'northern hemisphere', ', the', 'summer solstice', 'occurs in', 'June'], ['In the', 'southern hemisphere', ', the', 'summer solstice', 'occurs in', 'December'], ['In the', 'northern hemisphere', ', the', 'winter solstice', 'occurs in', 'December'], ['In the', 'southern hemisphere', ', the', 'winter solstice', 'occurs in', 'June'], ['In the', 'northern hemisphere', ', the', 'spring equinox', 'occurs in', 'March'], ['In the', 'southern hemisphere', ', the', 'spring equinox', 'occurs in', 'September'], ['In the', 'northern hemisphere', ', the', 'fall equinox', 'occurs in', 'September'], ['In the', 'southern hemisphere', ', the', 'fall equinox', 'occurs in', 'March']]}
        answers = pipe.RCI(query, table)
#         # print(answers[0]['row_ndx'])
#
        df = pd.DataFrame(table['rows'],columns=[table['header']])
        st.dataframe(df.style.applymap(lambda _: "background-color: CornflowerBlue;", subset=(answers[0]['row_ndx'],slice(None))))

# for answer in answers:
#     st.write(answer)





