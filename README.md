# Quasar
[![Build Status](https://circleci.com/gh/qcware/quasar.svg?style=svg&circle-token=e85544db6236d5ecb720ac042a9a40d2f819a4ec)](https://circleci.com/gh/qcware/quasar.svg?style=svg&circle-token=e85544db6236d5ecb720ac042a9a40d2f819a4ec)

## Environment setup:
Reccommened: create a virtual environment with virtualenv or conda e.g on linux:
```
virtualenv quasar-venv --python=python3.7
```
To use with jupyter notebooks
```
pip install ipykernel
ipython kernel install --user --name=quasar-venv
```
Install requirements with
```
pip install -r requirements.txt
```

## To use Forest + PyQuil:
1. Install Forest SDK (request here: https://www.rigetti.com/forest)<br/>
2. Run the following commands:
```
qvm -S 
quilc -S 
```
