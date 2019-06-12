#!/bin/bash

for ((i=1;i<=5;i++));
do
 wget http://pr.cs.cornell.edu/web3/CAD-120/data/Subject${i}_rgbd_images.tar.gz
 wget http://pr.cs.cornell.edu/humanactivities/data/Subject${i}_annotations.tar.gz
 wget http://pr.cs.cornell.edu/web3/CAD-120/data/Subject${i}_rgbd_rawtext.tar.gz
done
