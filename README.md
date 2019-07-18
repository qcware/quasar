# Quasar
[![Build Status](https://circleci.com/gh/qcware/quasar.svg?style=svg&circle-token=e85544db6236d5ecb720ac042a9a40d2f819a4ec)](https://circleci.com/gh/qcware/quasar.svg?style=svg&circle-token=e85544db6236d5ecb720ac042a9a40d2f819a4ec)

## Environment setup:
Reccommened: create a virtual environment with virtualenv or conda. E.g.:
```
virtualenv quasar-venv --python=python3.7
```
Then, activate your virtual environment (command to activate is OS dependent). <br>

Install requirements with:
```
pip install -r requirements.txt
```
To use with jupyter notebooks, inside your virtual environment run:
```
pip install ipykernel
ipython kernel install --user --name=quasar-venv
```
You may then have to change kernel inside the jupyter notebook you are working on.

## To use Forest + PyQuil:
1. Install Forest SDK (request here: https://www.rigetti.com/forest)<br/>
2. Run the following commands:
```
qvm -S 
quilc -S 
```
