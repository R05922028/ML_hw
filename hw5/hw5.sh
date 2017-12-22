#!/bin/bash
wget -O model_hw5_mf.h5 "https://www.dropbox.com/s/7fsui2p0dihpzkp/model_hw5_mf.h5?dl=1"
python3 hw5.py $1 $2 $3 $4
