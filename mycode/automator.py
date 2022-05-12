import itertools
import socket

import subprocess
from certify_utilis import get_dir, extract_summary_cifar, extract_summary_mnist
from notification import NOTIFIER
from datetime import datetime
from pathlib import Path


MODE = ['train', 'neval', 'certify', 'plot', 'nablation', 'nsub-acc-test', 'nsummary']
DATASET = 'cifar10'
TRAIN_MODE = 'Bagging' # DP, Sub-DP, Bagging, Sub-DP-no-amp

# No saving
TRAIN_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --train-mode {train_mode}' # --save-model

EVAL_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --train-mode {train_mode} --load-model'

CERTIFY_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --training-size {training_size} --sub-training-size {sub_training_size} --train-mode {train_mode} --mode {mode}'

TRAIN_SUBSET_ACC = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --save-model --train-mode {train_mode} --sub-acc-test'

if DATASET == 'mnist':
    results_folder = '../results/mnist'
    model_names = ['LeNet']
    training_size = 60000
    n_runss = [1]
    epochss = [1000]
    sigmas = [1.0] # (sigma, clip): (1.0, 2.4), (2.0, 1.3), (3.0, 0.8), (4.0, 0.7)
    sample_rates = [0.001]
    lrs = [0.1]
    clips = [1.0]
    sub_training_sizes = [50]


if DATASET == 'fashion_mnist':
    results_folder = '../results/fashion_mnist'
    model_names = ['LeNet']
    training_size = 60000
    n_runss = [1000]
    epochss = [1]
    sigmas = [3.0] # sigma=1.0, C=2.1; sigma=2.0, C=1.0; sigma=3.0, C=1.0; sigma=4.0, C=0.8; 
    sample_rates = [0.001]
    lrs = [0.1]
    clips = [1.0]
    sub_training_sizes = [500]


elif DATASET == 'cifar10':
    results_folder = '../results/cifar10'
    model_names = ['ResNet18-DP']
    training_size = 50000
    n_runss = [1000]
    epochss = [200]
    sigmas = [1.0] # sigmas = [1.0, 1.5, 2.0]
    sample_rates = [0.01024] # sample_rates = [512/10000, 1024/10000]
    lrs = [0.1] # lrs = [0.01, 0.05, 0.1]
    clips = [50] # clips = [34 for sigma=1]
    sub_training_sizes = [500]
    

if 'train' in MODE:
    for nr, ep, sig, sr, lr, c, sts, mn in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes, model_names): 
        cmd = TRAIN_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=mn, sub_training_size=sts, train_mode=TRAIN_MODE)
        print(cmd)
        subprocess.call(cmd.split())

if 'eval' in MODE:
    for nr, ep, sig, sr, lr, c, sts, mn in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes, model_names): 
        cmd = EVAL_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=mn, sub_training_size=sts, train_mode=TRAIN_MODE)
        print(cmd)
        subprocess.call(cmd.split())

if 'certify' in MODE:
    for nr, ep, sig, sr, lr, c, sts, mn in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes, model_names): 
        cmd = CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=mn, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE, mode='certify')
        print(cmd)
        subprocess.call(cmd.split())
        
if 'plot' in MODE:
    for nr, ep, sig, sr, lr, c, sts, mn in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes, model_names): 
        cmd = CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=mn, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE, mode='plot')
        print(cmd)
        subprocess.call(cmd.split())

if 'ablation' in MODE:
    for nr, ep, sig, sr, lr, c, sts, mn in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes, model_names): 
        cmd = CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=mn, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE, mode='ablation')
        print(cmd)
        subprocess.call(cmd.split())

if 'sub-acc-test' in MODE:
    for nr, ep, sig, sr, lr, c, sts, mn in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes, model_names): 
        cmd = TRAIN_SUBSET_ACC.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=mn, sub_training_size=sts, train_mode=TRAIN_MODE)
        print(cmd)
        subprocess.call(cmd.split())

if 'summary' in MODE:
    summarys = []
    for nr, ep, sig, sr, lr, c, sts, mn in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes, model_names):
        dir_path = get_dir(TRAIN_MODE, results_folder, mn, lr, sig, c, sr, ep, nr, sts)
        trainlog_path = f'{dir_path}/train.log'
        with open(trainlog_path, 'r') as f:
            lines = f.readlines()
            if DATASET == 'cifar10':
                acc, eps = extract_summary_cifar(lines)
            elif DATASET == 'mnist' or DATASET == 'fashion_mnist':
                acc, eps = extract_summary_mnist(lines)
            summarys.append((dir_path, acc, eps))
    summarys.sort(key=lambda x:x[1])
    
    # make folder 
    summary_folder = f'{results_folder}/summary_{datetime.now()}'
    Path(summary_folder).mkdir(parents=True, exist_ok=True)
    with open(f'{summary_folder}/summary_{datetime.now()}.txt', 'w') as f:
        for line in summarys:
            f.write(str(line) + '\n')
    # move dirs
    import shutil
    for nr, ep, sig, sr, lr, c, sts, mn in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes, model_names):
        ori_dir_path = get_dir(TRAIN_MODE, results_folder, mn, lr, sig, c, sr, ep, nr, sts)
        tar_dir_path = ori_dir_path.replace(results_folder, summary_folder)
        shutil.move(ori_dir_path, tar_dir_path)



NOTIFIER.notify(socket.gethostname(), 'Job Done.')