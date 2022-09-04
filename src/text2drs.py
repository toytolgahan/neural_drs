import random
import string
import pickle
from torch import linalg as LA
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import spacy

class encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, p):
        super(encoder, self).__init__()
        self.drop = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=p)
    def forward(self, x):
        embedding = self.drop(self.embed(x))
        seq_len = embedding.shape[-3]
        batch_size = embedding.shape[-2]
        hidden = torch.randn((self.num_layers,batch_size,self.hidden_size))
        cell = torch.randn(hidden.shape)
        outputs = torch.zeros(seq_len,1, batch_size, self.hidden_size)
        hiddens = torch.zeros(seq_len, self.num_layers, batch_size, self.hidden_size)
        cells   = torch.zeros(seq_len,self.num_layers, batch_size, self.hidden_size)
        embedding = embedding.squeeze(0)
        for n, input in enumerate(embedding):
            input = input.unsqueeze(0)
            output, (hidden, cell) = self.rnn(input, (hidden,cell))
            outputs[n] = output.unsqueeze(0)
            hiddens[n] = hidden
            cells[n] = cell
        return outputs, hiddens, cells
    
class decoder1(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers,p):
        super(decoder1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.drop = nn.Dropout(p)
        #ATTENTION
        self.attn = nn.Linear(self.hidden_size*2, 1)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        ####
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=p)

    def forward(self, x, hidden, cell, encoder_outputs, encoder_hidden, encoder_cell):
        embedding = self.drop(x)
        attn_weights = F.softmax(self.attn(torch.cat((embedding, encoder_hidden.unsqueeze(0)),-1)),dim=-1)
        encoder_outputs = encoder_outputs.squeeze(1)
        attn_applied = torch.mul(attn_weights, encoder_outputs)
        attn_applied = attn_applied.sum(0).unsqueeze(0)
        output = torch.cat((embedding, attn_applied), -1)
        output = self.attn_combine(output)
        output = F.relu(output)
        outputs, (hidden, cell) = self.rnn(output, (hidden, cell))
        return outputs, hidden, cell


class decoder2(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, p, vocab_size, encoder_seq_size):
        super(decoder2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(p)
        self.embed = nn.Embedding(self.vocab_size + 1, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=p)
        #ATTENTION
        self.attn = nn.Linear(self.embed_size*2, encoder_seq_size)
        self.attn_combine = nn.Linear(self.embed_size*2, self.embed_size)
        #####
        self.rnn = nn.LSTM(embed_size, embed_size, num_layers, dropout=p)
        self.out = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs, encoder_hidden, encoder_cell):
        x = self.embed(x)
        embedding = self.drop(x)
        attn_weights = F.softmax(self.attn(torch.cat((embedding, encoder_hidden), -1)),dim=-1)
        encoder_outputs = encoder_outputs.squeeze(1)
        attn_applied = torch.mul((attn_weights.T).unsqueeze(-1), encoder_outputs)
        attn_applied = attn_applied.sum(0).unsqueeze(0)
        embedding = embedding.unsqueeze(0)
        output = torch.cat((embedding, attn_applied), -1)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        output = F.softmax(self.out(output[0]), dim=-1)
        return output, hidden, cell
    
class seq2seq(nn.Module):
    def __init__(self, encoder, decoder1, decoder2):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.softmax = nn.Softmax(dim=2)
    def forward(self, source, target_size, dataset, pretraining):
        source = source.unsqueeze(0)
        batch_size = source.shape[-1]
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(source)
        hidden = encoder_hidden[-1]
        cell = encoder_hidden[-1]
        encoder_hidden = encoder_hidden[-1,-1,:,:]
        encoder_cell = encoder_cell[-1,-1,:,:]
        
        if pretraining:
            x = dataset.concepts[0][0]
            x = x.view(-1,1)
            x = x@torch.ones(1, encoder_hidden.shape[-2]) # hidden.shape[1] <-- batch size
            x = x.T
            x = x. unsqueeze(0)
            predictions = torch.zeros([target_size] + list(x.shape))
            for i in range(1, target_size):
                output, hidden, cell = self.decoder1(x, hidden, cell, encoder_outputs, encoder_hidden, encoder_cell)
                x = output
                predictions[i] = output
        else:
            predictions = torch.zeros(dataset[0][1][1].shape[0], len(charList), batch_size)
            x = torch.tensor([len(charList)]).long()
            x = x.repeat(batch_size)
            for i in range(predictions.shape[1]):
                output, hidden, cell = self.decoder2(x, hidden, cell, encoder_outputs, encoder_hidden, encoder_cell)
                x = torch.argmax(output, dim=1)
                predictions[i] = output.T
        return predictions
    
    
    
class Data(Dataset):
    def __init__(self):
        with open('../data/target_indexes.pickle', 'rb') as d:
            self.target_indexes = pickle.load(d)
        with open('../data/tokens.pickle', 'rb') as e:
            self.inp = pickle.load(e)
        with open('../data/embeddings.pickle', 'rb') as f:
            concepts = pickle.load(f)
        self.concepts = torch.Tensor(concepts)
        with open('../data/inputVocab.pickle', 'rb') as h:
            self.vocab = list(set(pickle.load(h)))
        with open('../data/index2word.pickle', 'rb') as g:
            self.index2word = pickle.load(g)
        word2index = {}
        for n, word in enumerate(self.vocab):
            word2index[word] = n
        self.word2index = word2index
        self.len = len(self.inp)

    def __getitem__(self,index):
        x = self.inp[index]
        x = torch.from_numpy(x)
        x = x.int()
        y = self.concepts[index], self.target_indexes[index]
        return x, y
    def __len__(self):
        return self.len
    def index2exp(self, indexlist):
        expressions = []
        for indexes in indexlist:
            wordsList = [self.index2word[i] if i in list(self.index2word.keys()) else "*" for i in indexes]
            expression = ' '.join(wordsList)
            expressions.append(expression)
        return expressions
    def exp2index(self, exp):
        tokens = nlp(exp)
        indexes = [self.word2index[token.text] if token.text in self.vocab else random.randint(1, len(self.vocab)-1) for token in tokens]
        return indexes
    

    

#DATA SPLIT
batch_size = 4
dataset= Data()
train_len = int(len(dataset)*(2/3))
test_len = len(dataset) - train_len
train_data, val_data = random_split(dataset, [train_len, test_len])

trainloader = DataLoader(dataset=train_data, batch_size=batch_size)

embed_dimEnc = 50
embed_dimDec = len(dataset.concepts[0][0])  #target vector dimension

charList = string.printable
encoder_seq_size = dataset[0][0].shape[0]

#THE MODEL
enc = encoder(len(dataset.vocab), embed_dimEnc, embed_dimDec, 3, 0.5)
dec1 = decoder1(embed_dimDec, embed_dimDec, 3, 0.5)
dec2 = decoder2(embed_dimDec, len(charList), 3, 0.5, len(charList), encoder_seq_size)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = seq2seq(enc, dec1, dec2)
try:
    model.load_state_dict(torch.load('../models/model.pth'))
    
except:
    print("no model is saved yet")
model.to(device)
#TRAINING
#Hyperparameters
learning_rate = 0.05
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CosineEmbeddingLoss()

epochs=25


#Train the model
def pre_train(epoch, learning_rate, optimizer, criterion):
    for epoch in range(epochs):
        LOSS = 0
        for n, (x,y) in enumerate(trainloader):
            y = y[0]
            x = torch.transpose(x, 0,1)
            y = torch.transpose(y, 0,1)
            yhat = model(x, y.shape[0], dataset, pretraining=True)
            yhat = yhat.squeeze(1)
            y = y.view(yhat.shape)
            optimizer.zero_grad()
            loss = criterion(torch.flatten(yhat), torch.flatten(y), torch.tensor(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            LOSS += loss
    print("in epoch {}, the loss is {}".format(epoch, LOSS))


pre_train(epochs, learning_rate, optimizer, criterion)


def train(epochs, learning_rate, optimizer, criterion):
    for epoch in range(epochs):
        LOSS = 0
        for n, (x, y) in enumerate(trainloader):
            y = y[1]
            x = torch.transpose(x, 0, 1)
            y = torch.transpose(y, 0, 1)
            yhat = model(x , y.shape[0], dataset, pretraining=False)
            yhat = yhat.squeeze(1)
            optimizer.zero_grad()
            loss = criterion(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            LOSS += loss
    print("in epoch {}, the loss is {}".format(epoch, LOSS))

    
criterion2 = nn.CrossEntropyLoss()
train(epochs, learning_rate, optimizer, criterion2)

torch.save(model.state_dict(), '../models/model.pth')

#TRANSLATION

def translate(x):
    expression = ""
    for n in x:
        exp += charList[n[0]]
    return expression
