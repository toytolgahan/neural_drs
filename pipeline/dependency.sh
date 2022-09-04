#!/bin/bash


sudo apt update

#install pip3
modules2=$(apt list --installed | grep pip3)
size2=${#modules2}
[[ size2 -eq 0 ]] && sudo apt install python3-pip || echo "pip3 is already installed"


#install numpy
modules3=$(apt list --installed | grep numpy)
size3=${#modules3}
[[ size3 -eq 0 ]] && pip3 install numpy || echo "numpy is already installed"


#install pytorch
modules4=$(apt list --installed | grep torch)
size4=${#modules4}
[[ size4 -eq 0 ]] && pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu || echo "torch is already installed"


#install spacy
modules5=$(apt list --installed | grep spacy)
size5=${#modules5}
[[ size5 -eq 0 ]] && pip install -U pip setuptools wheel && pip install -U spacy && python3 -m spacy download en_core_web_trf || echo "spacy is already installed -- $size5"
