import torch.nn.functional as F
import urllib
from newspaper import Article
import newspaper
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import nltk
from torch.utils.data import Dataset, DataLoader
import torch
import torch.onnx
import pickle
import os
import re

import keras_preprocessing
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
# from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as f
import pandas as pd
import json
import numpy as np
import os
DIRECTORY = 'data'
paths = []

for root, dirs, files in os.walk(DIRECTORY):
    for name in files:
        paths.append(os.path.join(root, name))

names = [i.split('/')[-1] for i in paths][1:]
data_dict = dict(zip([i[:-4] for i in names], paths[1:]))

class PreprocessingDataset(Dataset):
    def __init__(self, file, root, x_col, y_col, meta_columns, label_idx = -1):
        self.x_col = x_col
        self.y_col = y_col
        self.data = pd.read_csv(file)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data = self.data.drop(meta_columns, axis=1)

        self.x_data = self.data[x_col]
        self.max_len = max([len(i) for i in self.x_data])
        self.max_len = 600

        self.x_data, self.token = self.word_vector(self.x_data)

        self.data[x_col] = [torch.tensor(i) for i in self.x_data]
        self.data = self.vectorize(self.data, [y_col])
        self.df_data = self.data
        self.data = self.data.to_numpy()

        self.root = root

    def format_text(self, token):
        clean_token = ''.join(chr for chr in token if chr.isalnum() and chr.isalpha())
        return clean_token

    def word_vector(self, data):
        x_data = data
        x_data = list(x_data)
        maximum_length = 0
        max_idx = 0
        for idx, i in enumerate(x_data):

            if len(i) > maximum_length:
                maximum_length = len(i)
                max_idx = idx
        maximum_length = 600
        t = Tokenizer(num_words=600, filters='\n.,:!"#$()&@%^()-_`~[];.,{|}')
        t.fit_on_texts(x_data)
        sequences = t.texts_to_sequences(x_data)
        sequences = sequence.pad_sequences(sequences, maxlen=maximum_length)

        return sequences, t


    def vectorize(self, data_inp, columns):
        data = data_inp
        for column in columns:
            labels = list(data[column].unique())
            ref = dict(zip(data[column].unique(), [i for i in range(len(labels))]))
            data.loc[:, column] = [torch.tensor(ref[data.loc[idx, column]]) for idx in range(len(data))]

        return data

    def __len__ (self):
        return len(self.data)

    def __getitem__ (self, idx):
        
        self.transpose_data = self.data
        self.transpose_data = self.transpose_data.transpose()
        x_data = self.transpose_data[0]
        y_data = self.transpose_data[1]

        return x_data[idx], y_data[idx]

clean_truth_data = PreprocessingDataset(data_dict['politifact_clean_binarized'], DIRECTORY, 'statement', 'veracity', ['source', 'link'])
token_basis = clean_truth_data.token

BATCH_SIZE = 64

primary_data = clean_truth_data #secondary option of truth_data

train_len = int(len(primary_data)*0.8)
test_len = len(primary_data) - train_len

train_set, test_set = torch.utils.data.random_split(primary_data, [train_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

a = iter(train_loader)
b = np.array(next(a)[0])
inp_size = (b.shape)[1]
emb_dim = 6712800

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeedForward(nn.Module):
    def __init__(self, num_classes, input_size, kernel_size=4):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc3 = nn.Linear(200, 100)        
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 20)
        self.fc7 = nn.Linear(20, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))

        
        return x


class RecurrentClassifier(nn.Module):
    def __init__(self, embedding_dim, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super(RecurrentClassifier, self).__init__()

        self.embedding = nn.Embedding(embedding_dim, input_size)
        self.rnn = nn.LSTM(input_size, 
                            hidden_size,
                            num_layers,
                            batch_first = True,
                            dropout=dropout)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x, (hidden, cell) = self.rnn(x)
        print(hidden.shape)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1, :, :]), dim=1))
        x = self.fc1(hidden)
        x = self.dropout(self.fc2(x))

        return x

max_len = len(train_set[1][0])
ref_check = 1

feedforward = FeedForward(ref_check, inp_size).to(device)
# recurrent = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)


# with open('serialized/recurrent_empty.pickle', 'wb') as f:
#     recurrent = pickle.load(f)

# with open('serialized/feedforward_empty.pickle', 'wb') as f:
#     feedforward = pickle.load(f)

def train(net, train_loader, LR, DECAY, EPOCHS):
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=DECAY)
    loss_func = torch.nn.BCEWithLogitsLoss()

    epochs = EPOCHS
    losses = []

    for step in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inp, labels = data
            if net == recurrent:
                inp, labels = inp.long().to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = net(inp)
                cost = loss_func(torch.squeeze(outputs), torch.squeeze(labels))
            elif net == feedforward:
                inp, labels = inp.float().to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = net(inp)
                cost = loss_func(torch.squeeze(outputs), labels)
            cost.backward()
            optimizer.step()

            running_loss += cost.item()
        print(f'Epoch: {step}   Training Loss: {running_loss/len(train_loader)}')
    print('Training Complete')  

    return losses

def eval(net, test_loader):
    total = 0
    acc = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=DECAY)

    for i, data in enumerate(test_loader):
        inp, labels = data
        optimizer.zero_grad()
        output = net(inp.float())
        output = output.detach().numpy()
        output = list(output)
        output = [list(i).index(max(i)) for i in output]
        
        for idx, item in enumerate(torch.tensor(output)):
            total += 1
            if item == labels[idx]:
                acc += 1
    print(f'{acc/total*100}%')

def format_raw_text(token):
    token = token.replace(' ', 'uxd')
    clean_token = ''.join(chr for chr in token if chr.isalnum() and chr.isalpha())
    return clean_token

def model_load(net, PATH, name, export=True):
    if export:
        torch.save(net.state_dict(), PATH+name+'.pth')
        return PATH+name+'.pth'
    else:
        net.load_state_dict(torch.load(PATH + name + '.pth'))
        net.eval()
        return net

def sentiment(inp_text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(inp_text)
    
def meta_extract(url):
    article = Article(url)
    article.download()
    article.parse()
    article.download('punkt')
    article.nlp()
    return article.authors, article.publish_date, article.top_image, article.images, article.title, article.summary

def tokenize_sequence(text_inp, tokenizer):
    text_inp = text_inp.lower().split('\n')
    tokenizer.fit_on_texts(text_inp)
    sequences = tokenizer.texts_to_sequences(text_inp)
    sequences = [i if i!=[] else [0] for i in tokenizer.texts_to_sequences(text_inp)]
    sequences = [i[0] for i in sequences]
    pad_len =  [0]*int(inp_size - len(sequences))
    sequences += pad_len
    return torch.FloatTensor(sequences)[:600]

def prediction(inp, model):
    output = model(inp)
    return output

# model_load(recurrent, 'model_parameters/', 'lstm_politifact')
