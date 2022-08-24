#!/bin/bash

if [ ! -f ../data/groningen.zip ];
then
	curl -o ../data/groningen.zip https://gmb.let.rug.nl/releases/gmb-1.0.0.zip
	unzip -d ../data/ ../data/groningen.zip
else
	echo "Gröningen Meaning Bank is already downloaded!"
fi


