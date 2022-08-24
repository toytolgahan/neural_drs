#!/bin/bash
if [ -f "./main.sh" ]
then
	echo "main.sh exists"
else
	echo "file is not found"
fi

if [ -f "./text2drs.py" ]
then
	echo "text2drs is here"
else
	echo "text2drs is not here"
fi

if [ -f "../src/text2drs.py" ]
then
	echo "text2drs is there"
else
	echo "text2drs is not there"
fi

installToy () {
        modules=[ $(apt list --installed | grep $1) ]
        size=${#modules[@]}
        [[ size -eq 0 ]] && $2 || echo $modules
}
installToy python echo do it

