#!/bin/bash
trap "" SIGABRT


count=$(cat "./results/test_zone_th0.24_Norway_selfatt_lr0.001_ustep5.csv" | wc -l)

while [ $count -lt 201 ]
do
  if [ $count -eq 0 ]
  then
    python main.py --country="Norway";
  else
    python main.py --country="Norway" --load_ckpt=True
  fi
  count=$(cat "./results/test_zone_th0.24_Norway_selfatt_lr0.001_ustep5.csv" | wc -l)
done
