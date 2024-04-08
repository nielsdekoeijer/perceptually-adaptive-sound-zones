#!/bin/bash
FILENAME_B="dataset/music_02.wav"
FILENAME_D="dataset/music_01.wav"

FS=8000
TX=5.0
TV=0.08

for B_NUM in {01..05}
do
  for D_NUM in {01..05}
  do
    # Skip if B_NUM and D_NUM are the same
    if [ "$B_NUM" != "$D_NUM" ]; then
      FILENAME_B="dataset/music_$B_NUM.wav"
      FILENAME_D="dataset/music_$D_NUM.wav"
      (faketime "-2 years" python3 run_experiment.py --fs $FS --b_filename $FILENAME_B --d_filename $FILENAME_D --mode apvast --D0 0.0 --tx $TX --tv $TV)
    fi
  done
done

wait
