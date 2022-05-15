#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs CIFAR10 training with differential privacy.
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from certify_utilis import result_folder_path_generator, gen_sub_dataset
from sampler import FixedSizedUniformWithReplacementSampler, UniformWithReplacementSampler
from models import ResNet18
from opacus.validators import ModuleValidator


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def setup(args):
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "DistributedDataParallel device_ids and output_device arguments \
            only work with single-device GPU modules"
        )

    if sys.platform == "win32":
        raise NotImplementedError("Windows version of multi-GPU is not supported yet.")

    # Initialize the process group on a Slurm cluster
    if os.environ.get("SLURM_NTASKS") is not None:
        rank = int(os.environ.get("SLURM_PROCID"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        world_size = int(os.environ.get("SLURM_NTASKS"))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "7440"

        torch.distributed.init_process_group(
            args.dist_backend, rank=rank, world_size=world_size
        )

        logger.debug(
            f"Setup on Slurm: rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        return (rank, local_rank, world_size)

    # Initialize the process group through the environment variables
    elif args.local_rank >= 0:
        torch.distributed.init_process_group(
            init_method="env://",
            backend=args.dist_backend,
        )
        rank = torch.distributed.get_rank()
        local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()

        logger.debug(
            f"Setup with 'env://': rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        return (rank, local_rank, world_size)

    else:
        logger.debug(f"Running on a single GPU.")
        return (0, 0, 1)


def cleanup():
    torch.distributed.destroy_process_group()


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(args, model, train_loader, optimizer, privacy_engine, epoch, device, print_result=False, optimizer_bn=None):

    torch.cuda.empty_cache()
    gc.collect()

    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(tqdm(train_loader)):

        print(torch.cuda.memory_summary())
        
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

        if optimizer_bn is not None:
            optimizer_bn.step()
            optimizer_bn.zero_grad()

        # measure accuracy and record loss
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)

        losses.append(loss.detach().item())
        top1_acc.append(acc1)

    if print_result:
        if not args.train_mode == 'Bagging':
            epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                delta=args.delta,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            )
            logger.info(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
            )
        else:
            logger.info(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )


def test(args, model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.detach().item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)

    logger.info(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    return np.mean(top1_acc)


def pred(args, model, test_loader, device):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(device)
            output = model(images)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            preds_list.extend(preds)
    return np.array(preds_list)


def softmax(args, model, test_loader, device):
    model.eval()
    softmax_list = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(device)
            output = model(images)
            softmax = nn.Softmax(dim=1)(output).detach().cpu().numpy()
            softmax_list.extend(softmax)
    return np.array(softmax_list)


# flake8: noqa: C901
def main():

    args = parse_args()

    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)
    
    # folder path
    result_folder = result_folder_path_generator(args)
    print(f'Result folder: {result_folder}')
    models_folder = f"{result_folder}/models"
    Path(models_folder).mkdir(parents=True, exist_ok=True)

    # set logging file path
    fh = logging.FileHandler(f"{result_folder}/train.log", mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Sets `world_size = 1` if you run on a single GPU with `args.local_rank = -1`
    if args.local_rank != -1 or args.device != "cpu":
        rank, local_rank, world_size = setup(args)
        device = local_rank
    else:
        device = "cpu"
        rank = 0
        world_size = 1

    generator = None

    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(
        # augmentations + normalize if args.train_mode == 'Bagging' else normalize
        augmentations + normalize
    )
    test_transform = transforms.Compose(normalize)

    def gen_train_dataset_loader(or_sub_training_size=None):
        train_dataset = CIFAR10(
            root=args.data_root, train=True, download=True, transform=train_transform
        )
        sub_training_size = args.sub_training_size if or_sub_training_size is None else or_sub_training_size

        if args.train_mode == 'Sub-DP' or args.train_mode == 'Bagging':
            train_dataset = gen_sub_dataset(train_dataset, sub_training_size, True)
        
        if args.train_mode == 'DP' or args.train_mode == 'Sub-DP':
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=args.workers,
                generator=generator,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(train_dataset),
                    sample_rate=args.sample_rate,
                    generator=generator,
                ),
            )
        elif args.train_mode == 'Sub-DP-no-amp':
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=args.workers,
                generator=generator,
                batch_sampler=FixedSizedUniformWithReplacementSampler(
                    num_samples=len(train_dataset),
                    sample_rate=args.sample_rate,
                    train_size=sub_training_size,
                    generator=generator,
                ),
            )
        else:
            print('No Gaussian Sampler')
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=args.workers,
                generator=generator,
                batch_size=128,
            )
        return train_dataset, train_loader

    def gen_test_dataset_loader():
        test_dataset = CIFAR10(
            root=args.data_root, train=False, download=True, transform=test_transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=args.workers,
        )
        return test_dataset, test_loader

    # collect votes from all models
    test_dataset, test_loader = gen_test_dataset_loader()
    aggregate_result = np.zeros([len(test_dataset), 10 + 1], dtype=np.int32)
    aggregate_result_softmax = np.zeros([args.n_runs, len(test_dataset), 10 + 1], dtype=np.float32)
    acc_list = []

    # use this code for "sub_training_size V.S. acc"
    if args.sub_acc_test:
        sub_acc_list = []

    for run_idx in range(args.n_runs):
        # Pre-training stuff for each base classifier
        
        # Define the model
        if args.model_name == 'ResNet18-DP':
            model = ModuleValidator.fix(ResNet18()).to(device)
        elif args.model_name == 'ResNet18':
            model = ResNet18().to(device)
        else:
            exit(f'Model name {args.model_name} invaild.')

        # Use the right distributed module wrapper if distributed training is enabled
        if world_size > 1:
            if not args.train_mode == 'Bagging':
                if args.clip_per_layer:
                    model = DDP(model, device_ids=[device])
                else:
                    model = DPDDP(model)
            else:
                model = DDP(model, device_ids=[device])

        if args.optim == "SGD":
            if args.skip_bn:
                # TODO: Always have momory leakage: tried to give non-bn-params in optimizer, not working; tried to freeze non-bn-parames in training, not working; 
                # It seems like as long as I don't feed ALL params into PrivacyEngine Optimizer, it will cause memory leakage. 
                optimizer = optim.SGD(
                    [{'params': param} for name, param in model.named_parameters() if not ('bn' in name or 'shortcut.1' in name)],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
                optimizer_bn = optim.SGD(
                    [{'params': param} for name, param in model.named_parameters() if 'bn' in name or 'shortcut.1' in name],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
                optimizer_bn = None
        # elif args.optim == "RMSprop":
        #     optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        # elif args.optim == "Adam":
        #     optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError("Optimizer not recognized. Please check spelling")

        # use this code for "sub_training_size V.S. acc"
        if args.sub_acc_test:
            sub_training_size = int(50000 - 50000 / args.n_runs * run_idx)
            _, train_loader = gen_train_dataset_loader(sub_training_size)
        else:    
            _, train_loader = gen_train_dataset_loader()

        # make model DP if necessary
        privacy_engine = None
        if not args.train_mode == 'Bagging':
            if args.clip_per_layer:
                # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
                n_layers = len(
                    [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                )
                max_grad_norm = [
                    args.max_per_sample_grad_norm / np.sqrt(n_layers)
                ] * n_layers
            else:
                max_grad_norm = args.max_per_sample_grad_norm

            privacy_engine = PrivacyEngine()
            clipping = "per_layer" if args.clip_per_layer else "flat"
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=max_grad_norm,
                clipping=clipping,
                poisson_sampling=False,
            )

        # Training and testing
    
        model_pt_file = f"{models_folder}/model_{run_idx}.pt"
        # load the model if it exists
        if os.path.isfile(model_pt_file) or args.load_model:
            model.load_state_dict(torch.load(model_pt_file))
        # train the model
        else:
            epoch_acc_epsilon = []
            logger.info(f'Run:{run_idx}')
            for epoch in range(args.start_epoch, args.epochs + 1):
                if args.lr_schedule == "cos":
                    lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    if args.skip_bn:
                        for param_group in optimizer_bn.param_groups:
                            param_group["lr"] = lr

                print_result = True if run_idx == 0 else False
                train(
                    args, model, train_loader, optimizer, privacy_engine, epoch, device, print_result=print_result, optimizer_bn=optimizer_bn
                )

                if args.run_test:
                    logger.info(f'Epoch: {epoch}')
                    test(args, model, test_loader, device)

                if run_idx == 0:
                    logger.info(f'Epoch: {epoch}')
                    acc = test(args, model, test_loader, device)
                    if not args.train_mode == 'Bagging':
                        eps, _ = privacy_engine.accountant.get_privacy_spent(
                            delta=args.delta,
                            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                        )
                        epoch_acc_epsilon.append((acc, eps))
            # save only for the first model
            if run_idx == 0:
                np.save(f"{result_folder}/epoch_acc_eps", epoch_acc_epsilon)


            # Post-training stuff 

            # use this code for "sub_training_size V.S. acc"
            if args.sub_acc_test:
                sub_acc_list.append((sub_training_size, test(args, model, test_loader, device)))

            # save the DP related data
            if run_idx == 0 and not args.train_mode == 'Bagging':
                dp_epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                    delta=args.delta,
                    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                )
                rdp_history = privacy_engine.accountant.history
                
                logger.info(f"epsilon {dp_epsilon}, best_alpha {best_alpha}")
                np.save(f"{result_folder}/dp_epsilon", dp_epsilon)
                np.save(f"{result_folder}/rdp_history", rdp_history)

        if world_size > 1:
            cleanup()

        # save preds and model
        logger.info(f'run_idx:{run_idx}')
        aggregate_result[np.arange(0, len(test_dataset)), pred(args, model, test_loader, device)] += 1
        aggregate_result_softmax[run_idx, np.arange(0, len(test_dataset)), 0:10] = softmax(args, model, test_loader, device)
        acc_list.append(test(args, model, test_loader, device))
        if not args.load_model and args.save_model:
            torch.save(model.state_dict(), model_pt_file)

    # Finish trining all models, save results
    aggregate_result[np.arange(0, len(test_dataset)), -1] = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))[1]
    aggregate_result_softmax[:, np.arange(0, len(test_dataset)), -1] = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))[1]
    np.save(f"{result_folder}/aggregate_result", aggregate_result)
    np.save(f"{result_folder}/aggregate_result_softmax", aggregate_result_softmax)
    np.save(f"{result_folder}/acc_list", acc_list)

    # use this code for "sub_training_size V.S. acc"
    if args.sub_acc_test:
        np.save(f"{result_folder}/subset_acc_list", sub_acc_list)



def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size-test",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size for test dataset (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--sample-rate",
        default=0.04,
        type=float,
        metavar="SR",
        help="sample rate used for batch construction (default: 0.005)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint",
        help="path to save check points",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/stat/tensorboard",
        help="Where Tensorboard log will be stored",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )
    parser.add_argument(
        "--lr-schedule", type=str, choices=["constant", "cos"], default="cos"
    )

    parser.add_argument(
        "--device", type=str, default="gpu", help="Device on which to run the code."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank if multi-GPU training, -1 for single GPU training. Will be overriden by the environment variables if running on a Slurm cluster.",
    )

    parser.add_argument(
        "--dist_backend",
        type=str,
        default="gloo",
        help="Choose the backend for torch distributed from: gloo, nccl, mpi",
    )

    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level (default: 0)",
    )

    # New added args

    parser.add_argument(
        "--model-name",
        type=str,
        default="ConvNet",
        help="Name of the model structure",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default="../results/cifar10",
        help="Where CIFAR10 results is/will be stored",
    )
    parser.add_argument(
        "--sub-training-size",
        type=int,
        default=0,
        help="Size of bagging",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        default=False,
        help="Run test for the model (default: false)",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        default=False,
        help="Load model not train (default: false)",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="DP",
        help="Train mode: DP, Sub-DP, Bagging",
    )
    parser.add_argument(
        "--sub-acc-test",
        action="store_true",
        default=False,
        help="Test subset V.S. acc (default: false)",
    )
    parser.add_argument(
        "--skip-bn",
        action="store_true",
        default=False,
        help="skip batchnorm",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
