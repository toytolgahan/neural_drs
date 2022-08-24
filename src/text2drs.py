import random
import spacy
import pickle
from torch import linalg as LA
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        print("outputs {} ".format(outputs.shape))
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


class decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers,p,encoderLength):
        super(decoder, self).__init__()
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
        attn_weights = F.softmax(self.attn(torch.cat((embedding.repeat(encoder_hidden.shape[0],1,1), encoder_hidden),2)))
        encoder_outputs = encoder_outputs.squeeze(1)
        attn_applied = torch.mul(attn_weights, encoder_outputs)
        attn_applied = attn_applied.sum(0).unsqueeze(0)
        output = torch.cat((embedding, attn_applied), -1)
        output = self.attn_combine(output)
        output = F.relu(output)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        return outputs, hidden, cell


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.softmax = nn.Softmax(dim=2)
    def forward(self, source, target_size, target_embeddings):
        source = source.unsqueeze(0)
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(source)
        hidden = encoder_hidden[-1]
        cell = encoder_hidden[-1]
        encoder_hidden = encoder_hidden[:,-1,:,:]
        encoder_cell = encoder_cell[:,-1,:,:]
        x = target_embeddings[0][0]
        x = x.view(-1,1)
        x = x@torch.ones(1, encoder_hidden.shape[-2]) # hidden.shape[1] <-- batch size
        x = x.T
        x = x. unsqueeze(0)
        predictions = torch.zeros([target_size] + list(x.shape))
        for i in range(1, target_size):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs, encoder_hidden, encoder_cell)
            x = output
            predictions[i] = output
        return predictions


class Data(Dataset):
    def __init__(self):
        with open('../data/tokens.pickle', 'rb') as f:
            self.inp = pickle.load(f)
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
        y = self.concepts[index]
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
        indexes = [self.word2index[token] if token in self.vocab else random.randint(1, len(self.vocab)-1) for token in tokens]
        return indexes


dataset= Data()
#train_len = int(len(dataset)*(2/3))
train_len = int(len(dataset)*(2/3))
test_len = len(dataset) - train_len
train_data, val_data = random_split(dataset, [train_len, test_len])
trainloader = DataLoader(dataset=train_data, batch_size=2)

encoderInpLen = dataset[2][0].shape[0]

embed_dimEnc = 50
embed_dimDec = len(dataset.concepts[0][0])  #target vector dimension

#THE MODEL
enc = encoder(len(dataset.vocab), embed_dimEnc, embed_dimDec, 3, 0.5)
dec = decoder(embed_dimDec, embed_dimDec, 3, 0.5,encoderInpLen)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = seq2seq(enc, dec)
try:
	model.load_state_dict(torch.load('../models/model.pth', map_location="cuda:0"))
except:
	print("no model is saved yet")
model.to(device)
#TRAINING
#Hyperparameters
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CosineEmbeddingLoss()

epochs=30

#Train the model
for epoch in range(epochs):
    LOSS = 0
    yhat_previous = torch.ones(dataset[0:2][1].shape)
    yhatlist = [yhat_previous, yhat_previous]
    for n, (x,y) in enumerate(trainloader):
        x = torch.transpose(x, 0,1)
        y = torch.transpose(y, 0,1)
        yhat = model(x, y.shape[0], dataset.concepts)
        yhat = yhat.squeeze(1)
        y = y.view(yhat.shape)
        optimizer.zero_grad()
        loss = criterion(torch.flatten(yhat), torch.flatten(y), torch.tensor(1))
        yhat_prev1 = yhatlist[-1]
        yhat_prev2 = yhatlist[-2]
        loss +=2*criterion(torch.flatten(yhat), torch.flatten(yhat_prev1), torch.tensor(-1))
        loss +=2*criterion(torch.flatten(yhat), torch.flatten(yhat_prev2), torch.tensor(-1))
        loss = loss/5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        LOSS += loss
        yhatlist.append(yhat)
    print("in epoch {}, the loss is {}".format(epoch, LOSS))

torch.save(model.state_dict(), '../models/model.pth')

#TRANSLATION
concepts = torch.flatten(dataset.concepts, start_dim=0, end_dim=-2)


eps = 1e-6
absVal = lambda x: LA.norm(x.float(),dim=-1).unsqueeze(-1)
similarity = lambda x, y: x/(absVal(x) + eps)@torch.transpose((y/(absVal(y)+eps)),-1,-2)


sim = similarity(yhat,concepts)
simMatrix = torch.argmax(sim,-1)

y1 = torch.transpose(dataset[4][1],0,1)
yhat = model(x, y1.shape[0], dataset.concepts)
yhat = yhat.squeeze(1)

translation = dataset.index2exp((simMatrix.T).tolist())[1]
print(translation)



