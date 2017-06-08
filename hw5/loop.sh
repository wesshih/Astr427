#!/bin/bash


make clean
make pi


for M in {0..24}
do
  ./calcpi.out 10 $M
done

