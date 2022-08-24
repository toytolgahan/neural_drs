#!/bin/bash
mkdir ../models
mkdir ../data

echo downloading the data from Gröningen Meaning Bank...
bash download_data.sh
wait
echo checking the dependencies...
bash dependency.sh
wait
if [ ! -f "../data/text_data/text0.txt" ]
then
	echo creating the text data directory
	mkdir ../data/text_data
	python3 ../src/raw2data.py
	echo raw2data processed
else
	echo there is this directory
fi
wait
if [ ! -f "../data/embeddings.pickle" ] || [ ! -f "../data/index2word.pickle" ] || [ ! -f "../data/inputVocab.pickle" ] || [ ! -f "../tokens.pickle" ]
then
	python3 ../src/pickles_data.py
	echo pickles_data processed
fi
wait
python3 ../src/text2drs.py
