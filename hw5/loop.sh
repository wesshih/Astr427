#!/bin/bash


make clean
make pi


for N in {0..24}
do
  ./calcpi.out 10 $N
done

