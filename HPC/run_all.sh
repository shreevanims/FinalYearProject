#!/bin/bash

sizes=(128 256 512 1024)
filters=(32 64 128 512)
threads=(1 2 4 8 12)

for s in "${sizes[@]}"
do
  echo "Preparing image size $s"
  python3 load.py $s   # 🔥 generate correct image

  for f in "${filters[@]}"
  do
    for t in "${threads[@]}"
    do
      for r in 1 2 3
      do
        ./run $s $f $t >> results.txt
      done
    done
  done
done