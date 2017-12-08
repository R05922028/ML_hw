#!/bin/bash
wget -O model.h5 "https://www.dropbox.com/s/5zwk4d4ojrrt4is/model.h5?dl=1"
python3 RNN_test.py $1 $2
