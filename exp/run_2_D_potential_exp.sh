#!/bin/sh

for potential_function in "POT_1"  "POT_2"  "POT_3"  "POT_4"
do                          
echo "Starting potential function "$potential_function
python ../src/fit_flow.py \
       --OUT_DIR ../out/ \
       --N_ITERS 10000 \
       --LR 1e-2 \
       --POTENTIAL $potential_function \
       --N_FLOWS 32\
       --BATCH_SIZE 100\
       --MOMENTUM .1\
       --N_PLOT_SAMPLES 10000

done
