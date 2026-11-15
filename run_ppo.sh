#!/bin/bash


lst_lr=(0.0003 0.0001 0.001)
lst_wd=(0 0.1 0.5)
layer_norm=("" "no-")
benchmark="9x9-k16-1"

id=1
for lr in ${lst_lr[@]}; do
  for wd in ${lst_wd[@]}; do
    for ln in "${layer_norm[@]}"; do
      echo "--cfg.weight-decay ${wd} --cfg.actor-lr=${lr} --cfg.critic-lr=${lr} --benchmark=${benchmark} --cfg.${ln}layer-norm --reps=30" 
      # sbatch ppo_job.sh --cfg.weight-decay ${wd} --cfg.actor-lr=${lr} --cfg.critic-lr=${lr} --benchmark=${benchmark} --cfg.${ln}layer-norm --reps=30
      id=$((id + 1))    
    done
  done
done
echo "$id"
