#!/usr/bin/env bash
for framework in tensorflow caffe; do
 for boxsize in 96 128 192 320; do
  for gpu_flag in 0 1; do
   python testbench.py -v=curling.mp4 -f=$framework -b=$boxsize -g=$gpu_flag
  done
 done
done