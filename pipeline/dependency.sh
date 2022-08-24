#!/bin/bash


sudo apt update

#install Python3
modules=$(apt list --installed | grep python3)
size=${#modules[@]}
[[ size -eq 0 ]] && sudo apt install python3 || echo "python3 is already installed"



#install pip3
modules=$(apt list --installed | grep pip3)
size=${#modules[@]}
[[ size -eq 0 ]] && sudo apt install pip3 || echo "pip3 is already installed"


#install numpy
modules=$(apt list --installed | grep numpy)
size=${#modules[@]}
[[ size -eq 0 ]] && pip3 install numpy || echo "numpy is already installed"


#install pytorch
modules=$(apt list --installed | grep torch)
size=${#modules[@]}
[[ size -eq 0 ]] && pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu || echo "torch is already installed"


#install spacy
modules=$(apt list --installed | grep spacy)
size=${#modules[@]}
[[ size -eq 0 ]] && pip install -U pip setuptools wheel || echo "spacy is already installed"
[[ size -eq 0 ]] && pip install -U spacy || echo "spacy is already installed"
[[ size -eq 0 ]] && python3 -m spacy download en_core_web_trf || echo "spacy transformers is already installed"
