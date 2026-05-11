#!/bin/bash

# ---------------- PARAMETERS ----------------
sizes=(128 256 512 1024)
filters=(32 64 128 512)
threads=(1 2 4 8 12)

# Scheduling options
schedules=(static dynamic guided)
chunks=(2 4 8 16 32)

# ---------------- RUN ----------------
for s in "${sizes[@]}"
do
  echo "Preparing image for size $s"
  python3 load.py $s

  for f in "${filters[@]}"
  do
    for t in "${threads[@]}"
    do
      for sch in "${schedules[@]}"
      do
        for ch in "${chunks[@]}"
        do
          for r in 1 2 3
          do
            echo "Running: size=$s filters=$f threads=$t schedule=$sch chunk=$ch run=$r"
            ./run $s $f $t $sch $ch
          done
        done
      done
    done
  done
done

echo "✅ All experiments completed!"