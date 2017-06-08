#!/bin/bash

echo 'looping bitch!'

make clean
make pi


for N in {5..20}
do
  for M in {5..20}
  do
    echo "N: $N, M: $M"
    ./calcpi.out $N $M
  done
done

