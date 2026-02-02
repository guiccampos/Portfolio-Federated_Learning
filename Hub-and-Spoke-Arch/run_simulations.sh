#!/bin/bash
set -euo pipefail

source /home/g2campos/venv_master_thesis/bin/activate

###  Experiment Batch 1: original params, no modifications on local epochs, batch size and learning rate
declare -a iterations=(1)
# declare -a iterations=(1 2 3 4 5)
declare -a population_sizes=(13)
# declare -a population_sizes=(13 25 37 49 61)
declare -a datasets=("MNIST")
# declare -a datasets=("CIFAR10" "MNIST" "FASHION_MNIST" "SST2")
declare -a partitionings=("IID" "POW")
# declare -a partitionings=("IID" "POW")
declare -A seeds=([1]=123 [2]=456 [3]=789 [4]=1011 [5]=1213)
declare -A batch_sizes=(["MNIST"]=32
                        ["FASHION_MNIST"]=32 
                        ["CIFAR10"]=128 
                        ["SST2"]=32
                        ["CIFAR10_ResNet"]=64
                        ["CIFAR10_ShuffleNetV2"]=64 
                        ["CIFAR100_ResNet"]=64 
                        ["CIFAR100_ShuffleNetV2"]=64
                        )
                          
declare -A fl_rounds=(["MNIST"]=60
                      ["FASHION_MNIST"]=80
                      ["CIFAR10"]=200 
                      ["SST2"]=100 
                      ["CIFAR10_ResNet"]=100 
                      ["CIFAR10_ShuffleNetV2"]=100
                      ["CIFAR100_ResNet"]=100
                      ["CIFAR100_ShuffleNetV2"]=100
                      )

declare -A local_epochs=(["MNIST"]=5
                         ["FASHION_MNIST"]=5
                         ["CIFAR10"]=10
                         ["SST2"]=5
                         ["CIFAR10_ResNet"]=10
                         ["CIFAR10_ShuffleNetV2"]=10
                         ["CIFAR100_ResNet"]=10
                         ["CIFAR100_ShuffleNetV2"]=10
                         )

declare -A learning_rates=(["MNIST"]=0.1
                           ["FASHION_MNIST"]=0.002
                           ["CIFAR10"]=0.025
                           ["SST2"]=0.0001
                           ["CIFAR10_ResNet"]=0.01
                           ["CIFAR10_ShuffleNetV2"]=0.1
                           ["CIFAR100_ResNet"]=0.01
                           ["CIFAR100_ShuffleNetV2"]=0.01
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
        python3 simulation_launcher.py \
          --dataset $dataset \
          --partitioning $partitioning \
          --seed ${seeds[$i]} \
          --batch_size ${batch_sizes[$dataset]} \
          --rounds ${fl_rounds[$dataset]} \
          --local_epochs ${local_epochs[$dataset]} \
          --learning_rate ${learning_rates[$dataset]} \
          --centralized_evaluation "true" \
          --run_id $i --output_dir "results"
      done
    done
  done
done 
