#!/bin/bash
set -euo pipefail
mkdir -p logs

source /home/g2campos/venv_master_thesis/bin/activate

# Setting 1: Learning rate decay configured to 0.99
declare -a iterations=(1 2 3 4 5)
declare -a population_sizes=(11)
declare -a datasets=("MNIST" "FASHION_MNIST" "Moderate_CIFAR10" "SST2" "SVHN")
declare -a partitionings=("IID" "POW")
declare -A seeds=([1]=1111 [2]=2222 [3]=3333 [4]=4444 [5]=5555)
declare -A batch_sizes=(["MNIST"]=32
                        ["FASHION_MNIST"]=32
                        ["Moderate_CIFAR10"]=64
                        ["SVHN"]=64
                        ["SST2"]=256
                        )
                          
declare -A fl_rounds=(["MNIST"]=30
                      ["FASHION_MNIST"]=40
                      ["Moderate_CIFAR10"]=100      # I was testing for 100 rounds
                      ["SVHN"]=50
                      ["SST2"]=50
                      )
                      
declare -A local_epochs=(["MNIST"]=1
                         ["FASHION_MNIST"]=1
                         ["Moderate_CIFAR10"]=1
                         ["SVHN"]=1                 # Should be 10 for FL
                         ["SST2"]=1
                         )

declare -A learning_rates=(["MNIST"]=0.015
                           ["FASHION_MNIST"]=0.005
                           ["Moderate_CIFAR10"]=0.02
                           ["SVHN"]=0.01
                           ["SST2"]=0.0001
                           )

for dataset in "${datasets[@]}"; do
  for population in "${population_sizes[@]}"; do 
    for partitioning in "${partitionings[@]}"; do
      for i in "${iterations[@]}"; do
        mpiexec -np $population \
        --map-by numa:pe=1 \
        --bind-to core \
        --rank-by numa \
        --report-bindings \
        python3 standalone_simulation_launcher.py \
          --dataset $dataset \
          --partitioning $partitioning \
          --seed ${seeds[$i]} \
          --batch_size ${batch_sizes[$dataset]} \
          --rounds ${fl_rounds[$dataset]} \
          --local_epochs ${local_epochs[$dataset]} \
          --learning_rate ${learning_rates[$dataset]} \
          --lr_gamma 0.99 \
          --centralized_evaluation "true" \
          --run_id $i --output_dir "results_setting_1"
      done
    done
  done
done 

declare -a datasets=("MNIST" "FASHION_MNIST" "Moderate_CIFAR10" "SVHN")
declare -a partitionings=("CLA")
for dataset in "${datasets[@]}"; do
  for population in "${population_sizes[@]}"; do 
    for partitioning in "${partitionings[@]}"; do
      for i in "${iterations[@]}"; do
        mpiexec -np $population \
        --map-by numa:pe=1 \
        --bind-to core \
        --rank-by numa \
        --report-bindings \
        python3 standalone_simulation_launcher.py \
          --dataset $dataset \
          --partitioning $partitioning \
          --seed ${seeds[$i]} \
          --batch_size ${batch_sizes[$dataset]} \
          --rounds ${fl_rounds[$dataset]} \
          --local_epochs ${local_epochs[$dataset]} \
          --learning_rate ${learning_rates[$dataset]} \
          --lr_gamma 0.99 \
          --centralized_evaluation "true" \
          --run_id $i --output_dir "results_setting_1"
      done
    done
  done
done 