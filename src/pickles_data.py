import spacy
import os
import numpy as np
import torch
import re
import pickle

nlp=spacy.load("en_core_web_trf")
dataPath = r"../data/text_data"

wordparts = lambda x, tensorData: tensorData.align[x].data.flatten()
tensors = lambda tensorData: (tensorData.tensors[0].reshape(-1, tensorData.tensors[0].shape[-1]))

wordVec = lambda x, tensorData: np.mean(tensors(tensorData)[wordparts(x, tensorData)], axis=0)

pattern = re.compile('\s+')

embeds = []
wordIndex = 0
index2word = {}
files = os.listdir(dataPath)
numOfdocs = int(len(files)/80)-1

for num in range(numOfdocs):
	text = "text"+str(num)
	drs = "drs"+str(num)
	filePath = os.path.join(dataPath, text)
	textContent = open(filePath, 'r', encoding='utf-8').read()
	textContent = re.sub(pattern, ' ', textContent)
	content = textContent
	content += " "
	filePath = os.path.join(dataPath, drs)
	drsContent = open(filePath, 'r', encoding='utf-8').read()
	drsContent = re.sub(pattern, ' ', drsContent) + "."
	content += drsContent
	content = re.sub(pattern, ' ', content)
	doc = nlp(content)
	tensorData = doc._.trf_data
	ts = tensors(tensorData)[0].reshape(-1,768)
	textLen = len(nlp(textContent))

	wordVecs = []
	for n, word in enumerate(doc):
		if n>= textLen:
			#WE NEED TO START FROM THE DRS PART. TEXT PART IS NEEDED FOR THE CONTEXTUAL BACKUP
			try:
				parts = wordparts(n, tensorData)
				wordEmbedding = wordVec(n, tensorData)
				wordVecs.append(wordEmbedding)
				index2word[wordIndex] = word.text
				wordIndex += 1
			except:
				print("the problem is the word {} ".format(word))
	embeds.append(wordVecs)

drsLengths = map(lambda x: len(x), embeds)
maxLength = max(drsLengths)
drsVectors = torch.zeros([len(embeds), maxLength, len(embeds[0][0])])
for n, drs in enumerate(embeds):
	for m, token in enumerate(drs):
		drsVectors[n][m]=torch.from_numpy(token)
with open('../data/embeddings.pickle', 'wb') as f:
	pickle.dump(drsVectors, f)
with open('../data/index2word.pickle', 'wb') as g:
	pickle.dump(index2word, g)

#INPUT DATA
content = []
texts = []
for num in range(numOfdocs):
	text = "text"+str(num)
	filePath = os.path.join(dataPath, text)
	textContent = open(filePath, 'r', encoding='utf-8').read()
	textContent = re.sub(pattern, ' ', textContent)
	textTokens = [token.text for token in nlp(textContent)]
	texts.append(textTokens)
	content += textTokens

vocab = list(set(content))

textLens = list(map(lambda x: len(x), texts))
maxLength = max(textLens)
textsTokenized = np.zeros((len(texts), maxLength))
for n, text in enumerate(texts):
	for m, word in enumerate(text):
		textsTokenized[n][m] = vocab.index(word)

with open('../data/tokens.pickle', 'wb') as f:
	pickle.dump(textsTokenized, f)
with open('../data/inputVocab.pickle', 'wb') as g:
	pickle.dump(vocab, g)

