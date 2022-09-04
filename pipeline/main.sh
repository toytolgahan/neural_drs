#!/bin/bash
#create a virtual environment
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate


mkdir ../models
mkdir ../data

echo downloading the data from Gröningen Meaning Bank...
bash download_data.sh
wait
echo checking the dependencies...
bash dependency.sh
wait

wait
if [ ! -f "../data/text_data/text0" ]
then
	echo creating the text data directory
	mkdir ../data/text_data
	python3 ../src/raw2data.py
	echo raw2data processed
else
	echo data already exits
fi
wait
if [ ! -f "../data/embeddings.pickle" ] || [ ! -f "../data/index2word.pickle" ] || [ ! -f "../data/inputVocab.pickle" ] || [ ! -f "../tokens.pickle" ]
then
	python3 ../src/pickles_data.py
	echo pickles_data processed
fi
wait
python3 ../src/text2drs.py
