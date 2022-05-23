#!/bin/bash
#SBATCH --job-name=task_1
#SBATCH --output=task_1.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:05:00
#SBATCH --mem=12G
#SBATCH --partition=shortgpgpu
#SBATCH --gres=gpu:4
#SBATCH -q gpgpumse
#SBATCH -A punim1629
#SBATCH --mail-user=404notfxxkingfound@gmail.com
#SBATCH --mail-type=ALL


echo "Loading required modules"
module load fosscuda/2020b
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0

echo "Loaded statsmodels" 
# pip3 install --upgrade pip
# pip3 install --user torch torchvision torchaudio
# pip3 install --user foolbox
# pip3 install --user git+https://github.com/fra31/auto-attack
# pip3 install --user scikit-learn
# pip3 install --user matplotlib

pip3 install statsmodels
# pip3 install -e .
pip3 install --user tqdm
pip3 install --user scikit-learn
pip3 install --user matplotlib

echo "Good to go!"
cd mycode
# python3 automator.py
# python3 -m torch.distributed.launch --nproc_per_node=4 cifar10.py --n-runs 5 --epochs 200 --sigma 0.0 --sample-rate 0.0256 --lr 0.1 -c 1000000 --model-name ResNet18 --sub-training-size 10000 --train-mode Sub-DP --batch-size-test 100 --dist_backend nccl
python3 -m torch.distributed.launch --nproc_per_node=4 automator.py