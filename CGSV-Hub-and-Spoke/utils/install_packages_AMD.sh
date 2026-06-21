#!/bin/bash

cd
python3 -m venv venv_master_thesis/
source venv_master_thesis/bin/activate

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
pip install numpy    
pip install matplotlib
pip install flwr
pip install flwr-datasets[vision]
pip install datasets
pip install matplotlib
pip install networkx
pip install pandas
pip install mpi4py
pip install scipy
pip install psutil

python - <<'PY'
import torch
print("torch:", torch.__version__, "HIP:", torch.version.hip)
print("cuda.is_available():", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
if torch.cuda.device_count(): print(torch.cuda.get_device_name(0))
PY