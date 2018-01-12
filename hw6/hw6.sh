#!/bin/bash
wget -O img_cluster.h5 "https://www.dropbox.com/s/05wucyioi32tzx3/img_cluster.h5?dl=1"
python3 image_clustering.py $1 $2 $3
