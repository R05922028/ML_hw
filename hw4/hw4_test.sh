#!/bin/bash
wget "https://www.dropbox.com/s/5zwk4d4ojrrt4is/model.h5?dl=0" -O model.h5
python3 RNN_test.py $1 $2
