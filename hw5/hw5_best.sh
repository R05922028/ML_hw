#!/bin/bash
wget -O model_hw5.h5 "https://www.dropbox.com/s/3rg5k2skt7u14ah/model_hw5.h5?dl=1"
python3 hw5_dnn.py $1 $2 $3 $4
