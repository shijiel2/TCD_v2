from __future__ import print_function
import numpy as np
from statsmodels.stats.proportion import (
    proportion_confint,
    multinomial_proportions_confint,
)
import argparse
import math
import os
import logging

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import scipy.stats
from certify_utilis import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha",
    type=float,
    default=0.001,
    metavar="SR",
    help="alpha used in estimate prob bar",
)
parser.add_argument(
    "-sr",
    "--sample-rate",
    type=float,
    default=0.001,
    metavar="SR",
    help="sample rate used for batch construction (default: 0.001)",
)
parser.add_argument(
    "-n",
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 14)",
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
    "--lr",
    type=float,
    default=0.1,
    metavar="LR",
    help="learning rate (default: .1)",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    metavar="S",
    help="Noise multiplier (default 1.0)",
)
parser.add_argument(
    "-c",
    "--max-per-sample-grad_norm",
    type=float,
    default=1.0,
    metavar="C",
    help="Clip per-sample gradients to this norm (default 1.0)",
)
parser.add_argument(
    "--results-folder",
    type=str,
    default="../results/mnist",
    help="Where MNIST results is/will be stored",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="SampleConvNet",
    help="Name of the model",
)
parser.add_argument(
    "--mode",
    type=str,
    default="certify",
    help="mode of the file",
)
parser.add_argument(
    "--training-size",
    type=int,
    default=60000,
    help="Size of training set",
)
parser.add_argument(
    "--train-mode",
    type=str,
    default="DP",
    help="Name of the methods: DP, Sub-DP, Bagging",
)
parser.add_argument(
    "--radius-range",
    type=int,
    default=70,
    help="Size of training set",
)
parser.add_argument(
    "--sub-training-size",
    type=int,
    default=30000,
    help="Size of training set",
)
args = parser.parse_args()


def certified_acc_against_radius(certified_poisoning_size_array, radius_range=50):
    certified_radius_list = list(range(radius_range))
    certified_acc_list = []

    for radius in certified_radius_list:
        certified_acc_list.append(
            len(
                certified_poisoning_size_array[
                    np.where(certified_poisoning_size_array >= radius)
                ]
            )
            / float(num_data)
        )
    return certified_acc_list, certified_radius_list


def certified_acc_against_radius_dp_baseline(clean_acc_list, dp_epsilon, dp_delta=1e-5, radius_range=50):
    _, est_clean_acc, _, _ = single_ci(clean_acc_list, args.alpha)
    # est_clean_acc = sum(clean_acc_list) / len(clean_acc_list)
    c_bound = 1
    def dp_baseline_certified_acc(k):
        p1 = np.e**(-k*dp_epsilon)*(est_clean_acc + (dp_delta*c_bound)/(np.e**(dp_epsilon)-1))-(dp_delta*c_bound)/(np.e**(dp_epsilon)-1)
        p2 = 0
        return max(p1, p2)
    
    certified_radius_list = list(range(radius_range))
    certified_acc_list = []
    for k in range(radius_range):
        certified_acc_list.append(dp_baseline_certified_acc(k))
    return certified_acc_list, certified_radius_list


def plot_certified_acc(c_acc_lists, c_rad_lists, name_list, color_list, linestyle_list, plot_path, xlabel='Radius', ylabel='Certified Accuracy'):
    print(plot_path)
    for c_acc_list, c_rad_list, name, color, linestyle in zip(c_acc_lists, c_rad_lists, name_list, color_list, linestyle_list):
        logger.info(f'(Rad, Acc):{list(zip(c_rad_list, c_acc_list))}')
        plt.plot(c_rad_list, c_acc_list, color, label=name, linewidth=1, linestyle=linestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()


def plot_interval(rad, lower, upper, xrange, plot_path, ylim=None):
    fig, ax = plt.subplots()
    ax.bar(xrange, upper, width=0.7, label='p1')
    ax.bar(xrange, lower, width=0.7, label='p2')
    ax.scatter(xrange, rad, s=0.7, label='rad', marker="s")
    if ylim:
        plt.ylim(ylim)
    ax.legend()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()


def certify(method_name):
    if 'softmax' not in method_name:
        gt, pred = aggres_meta_info(aggregate_result)
        logger.info(f"Clean acc: {(gt == pred).sum() / len(pred)}")
    else:
        aggregate_result_softmax_rm = np.mean(aggregate_result_softmax, axis=0)  
        gt, pred = aggres_meta_info(aggregate_result_softmax_rm)
        logger.info(f"Clean acc: {(gt == pred).sum() / len(pred)}")

    certified_poisoning_size_array = np.zeros([num_data], dtype=np.int32)
    dp_bagging_rads = []

    # if 'softmax' in method_name:
    #     import pickle
    #     mgf_diff_list = pickle.load(open('../results/mgf_diff_list.p', 'rb'))

    for idx in tqdm(range(num_data)):
        # Multinomial or Softmax scores
        if 'softmax' not in method_name:
            CI, ls = confident_interval_multinomial(aggregate_result, idx, method_name, float(args.alpha))
        else:
            CI, ls, varis = confident_interval_softmax(aggregate_result_softmax, aggregate_result_softmax_rm, idx, method_name, float(args.alpha))

        if method_name == 'dp' or method_name == 'dp_softmax':
            rd = CertifyRadiusDP(args, ls, CI, dp_epsilon, 1e-5)
        elif method_name == 'dp_baseline_size_one':
            rd = CertifyRadiusDP_baseline(args, ls, CI, dp_epsilon, 1e-5)
        elif method_name == 'rdp':
            rd = CertifyRadiusRDP(args, ls, CI,
                                  rdp_steps, args.sample_rate, args.sigma)
        elif method_name == 'rdp_softmax':
            rd = CertifyRadiusRDP(args, ls, CI,
                                  rdp_steps, args.sample_rate, args.sigma, softmax=True)
        # elif method_name == 'rdp_softmax_moments':
        #     rd = CertifyRadiusRDP_moments(args, ls, CI,
        #                           rdp_steps, args.sample_rate, args.sigma, mgf_diff_list, varis, softmax=True)
        elif method_name == 'rdp_gp':
            rd = CertifyRadiusRDP_GP(args, ls, CI,
                                  rdp_steps, args.sample_rate, args.sigma)
        elif method_name == 'best':
            rd1 = CertifyRadiusDP(args, ls, CI, dp_epsilon, 1e-5)
            rd2 = CertifyRadiusRDP(
                args, ls, CI, rdp_steps, args.sample_rate, args.sigma)
            rd = max(rd1, rd2)
        elif method_name == 'bagging':
            rd = CertifyRadiusBS(ls, CI, args.sub_training_size, args.training_size)
        elif method_name == 'dp_bagging':
            rd, dp_rad = CertifyRadiusDPBS(args, ls, CI, args.sub_training_size, args.training_size, dp_epsilon, 1e-5, rdp_steps, args.sample_rate, args.sigma)
            dp_bagging_rads.append(dp_rad)
        elif method_name == 'dp_bagging_softmax':
            rd, dp_rad = CertifyRadiusDPBS(args, ls, CI, args.sub_training_size, args.training_size, dp_epsilon, 1e-5, rdp_steps, args.sample_rate, args.sigma, softmax=True)
            dp_bagging_rads.append(dp_rad)
        elif method_name == 'dp_bagging_softmax_prob':
            rd = CertifyRadiusDPBS_softmax_prob(ls, CI, args.sub_training_size, args.training_size, 1e-5, rdp_steps, args.sample_rate, args.sigma)
        else:
            logging.warn(f'Invalid certify method name {method_name}')
            exit(1)
        certified_poisoning_size_array[idx] = rd
        # print('radius:', rd)
        # exit()

    if method_name == 'dp_bagging':
        np.save(f"{result_folder}/dp_bagging_rads.npy", dp_bagging_rads)
    elif method_name == 'dp_bagging_softmax':
        np.save(f"{result_folder}/dp_bagging_softmax_rads.npy", dp_bagging_rads)

    certified_acc_list, certified_radius_list = certified_acc_against_radius(
        certified_poisoning_size_array)

    logger.info(f'Clean acc: {(gt == pred).sum() / len(pred)}')
    logger.info(
        f'{method_name}: certified_poisoning_size_list:\n{certified_radius_list}')
    logger.info(
        f'{method_name}: certified_acc_list_dp:\n{certified_acc_list}')
    return certified_poisoning_size_array


if __name__ == "__main__":
    # main folder
    result_folder = result_folder_path_generator(args)
    print(result_folder)

    # set logging file path
    fh = logging.FileHandler(f"{result_folder}/certify.log", mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # laod data
    aggregate_result = np.load(f"{result_folder}/aggregate_result.npy")
    try:
        aggregate_result_softmax = np.load(f"{result_folder}/aggregate_result_softmax.npy")
    except:
        aggregate_result_softmax = None

    num_class = aggregate_result.shape[1] - 1
    num_data = aggregate_result.shape[0]

    if args.train_mode in ['DP', 'Sub-DP', 'Sub-DP-no-amp']:
        dp_epsilon = np.load(f"{result_folder}/dp_epsilon.npy")
        rdp_alphas = np.load(f"{result_folder}/rdp_alphas.npy")
        rdp_epsilons = np.load(f"{result_folder}/rdp_epsilons.npy")
        rdp_steps = np.load(f"{result_folder}/rdp_steps.npy")

        # log params
        logger.info(
            f'lr: {args.lr} sigma: {args.sigma} C: {args.max_per_sample_grad_norm} sample_rate: {args.sample_rate} epochs: {args.epochs} n_runs: {args.n_runs}')
        logger.info(f'dp  epsilon: {dp_epsilon}')
        logger.info(f'rdp epsilons: {rdp_epsilons}')
        logger.info(f'rdp steps: {rdp_steps}')
        logger.info(f'aggregate results:\n{aggregate_result}')
    else:
        logger.info(f'aggregate results:\n{aggregate_result}')

    # Certify
    if args.mode == 'certify':
        if args.train_mode in ['DP', 'Sub-DP', 'Sub-DP-no-amp']:
            np.save(f"{result_folder}/dp_cpsa.npy", certify('dp'))
            np.save(f"{result_folder}/rdp_cpsa.npy", certify('rdp'))    
            # np.save(f"{result_folder}/rdp_gp_cpsa.npy", certify('rdp_gp'))  
            # np.save(f"{result_folder}/dp_baseline_size_one_cpsa.npy", certify('dp_baseline_size_one'))
            # np.save(f"{result_folder}/best_dp_cpsa.npy", certify('best'))
            np.save(f"{result_folder}/dp_softmax_cpsa.npy", certify('dp_softmax'))
            np.save(f"{result_folder}/rdp_softmax_cpsa.npy", certify('rdp_softmax'))
            # np.save(f"{result_folder}/rdp_softmax_moments_cpsa.npy", certify('rdp_softmax_moments'))
            # if args.train_mode == 'Sub-DP':
            #     np.save(f"{result_folder}/dp_bagging_cpsa.npy", certify('dp_bagging'))
            #     np.save(f"{result_folder}/dp_bagging_softmax_cpsa.npy", certify('dp_bagging_softmax'))
        elif args.train_mode == 'Bagging':
            np.save(f"{result_folder}/bagging_cpsa.npy", certify('bagging'))

    
    # Ablation
    elif args.mode == 'ablation':
        test_size = 100

        # method_name = 'bagging'
        # p1_list, p2_list, rad_list = p1_p2_rad(test_size, np.load(f"{result_folder}/aggregate_result_bagging.npy"), np.load(f"{result_folder}/bagging_cpsa.npy"), method_name, args.alpha)
        # plot_interval(rad_list, p2_list, p1_list, range(test_size), f"{result_folder}/{method_name}_p1_p2_interval.png", ylim=[0,13])

        # method_name = 'dp_bagging'
        # p1_list, p2_list, rad_list = p1_p2_rad(test_size, aggregate_result, np.load(f"{result_folder}/dp_bagging_cpsa.npy"), method_name, args.alpha)
        # plot_interval(rad_list, p2_list, p1_list, range(test_size), f"{result_folder}/{method_name}_p1_p2_interval.png", ylim=[0,13])

        # method_name = 'dp_bagging_softmax'
        # p1_list, p2_list, rad_list = p1_p2_rad(test_size, aggregate_result_softmax, np.load(f"{result_folder}/dp_bagging_softmax_cpsa.npy"), method_name, args.alpha, aggregate_result_rm=np.mean(aggregate_result_softmax, axis=0))
        # plot_interval(rad_list, p2_list, p1_list, range(test_size), f"{result_folder}/{method_name}_p1_p2_interval.png", ylim=[0,13])

        # method_name = 'dp'
        # p1_list, p2_list, rad_list = p1_p2_rad(test_size, aggregate_result, [0]*num_data, method_name, args.alpha)
        # plot_interval(rad_list, p2_list, p1_list, range(test_size), f"{result_folder}/{method_name}_p1_p2_norad_interval.png", ylim=[0,1])

        # method_name = 'rdp_softmax'
        # p1_list, p2_list, rad_list = p1_p2_rad(test_size, aggregate_result_softmax, [0]*num_data, method_name, args.alpha, aggregate_result_rm=np.mean(aggregate_result_softmax, axis=0))
        # plot_interval(rad_list, p2_list, p1_list, range(test_size), f"{result_folder}/{method_name}_p1_p2_norad_interval.png", ylim=[0,1])
        
        # method_name = 'dp'
        # p1_list, p2_list, rad_list = p1_p2_rad(test_size, aggregate_result, np.load(f"{result_folder}/dp_cpsa.npy"), method_name, args.alpha)
        # plot_interval(rad_list, p2_list, p1_list, range(test_size), f"{result_folder}/{method_name}_p1_p2_interval.png")

        # method_name = 'dp_softmax'
        # p1_list, p2_list, rad_list = p1_p2_rad(test_size, aggregate_result_softmax, np.load(f"{result_folder}/dp_softmax_cpsa.npy"), method_name, args.alpha, aggregate_result_rm=np.mean(aggregate_result_softmax, axis=0))
        # plot_interval(rad_list, p2_list, p1_list, range(test_size), f"{result_folder}/{method_name}_p1_p2_interval.png")

        valid_radius = np.load(f"{result_folder}/rdp_softmax_ablation_valid_radius.npy")
        r_list, a_list, l_list, u_list = [], [], [], []
        for radius, alpha, delta, lower, upper in valid_radius:
            r_list.append(radius)
            a_list.append(alpha)
            l_list.append(lower)
            u_list.append(upper)
        plot_interval([0]*len(r_list), u_list, l_list, a_list, f"{result_folder}/rdp_softmax_lower_upper_interval.png")
        # print(r_list)
        
        
    # Plot
    elif args.mode == 'plot':

        # method_name = ['RDP-multinomial', 'RDP-softmax', 'ADP-multinomial', 'ADP-softmax', 'Baseline-DP', 'Baseline-Bagging']
        # method_name = ['RDP-multinomial', 'RDP-softmax', 'ADP-multinomial', 'ADP-softmax', 'Baseline-DP']
        # method_name = [r'$\sigma = 1.0$', r'$\sigma = 2.0$', r'$\sigma = 3.0$', r'$\sigma = 4.0$']
        # method_name = ['ResNet18-500', 'ResNet18-3000', 'ResNet18-DP-500', 'ResNet18-DP-3000']
        method_name = ['Baseline-Bagging']

        if args.train_mode in ['DP', 'Sub-DP', 'Sub-DP-no-amp', 'Bagging']:
            acc_list = []
            rad_list = []
            color_list = []
            linestyle_list = []
            for name in method_name:
                if name == 'ADP-multinomial':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/dp_cpsa.npy"), radius_range=args.radius_range)
                    col = 'tab:orange'
                    sty = 'solid'
                elif name == 'RDP-multinomial':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_cpsa.npy"), radius_range=args.radius_range)
                    col = 'tab:blue'
                    sty = 'solid'
                elif name == 'RDP-softmax':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_softmax_cpsa.npy"), radius_range=args.radius_range)
                    col = 'tab:blue'
                    sty = 'dashed'
                elif name == 'RDP-softmax-moments':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_softmax_moments_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'ADP-softmax':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/dp_softmax_cpsa.npy"), radius_range=args.radius_range)
                    col = 'tab:orange'
                    sty = 'dashed'
                elif name == 'Baseline-RDP-GP':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_gp_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'Baseline-DP':
                    acc, rad = certified_acc_against_radius_dp_baseline(np.load(f"{result_folder}/acc_list.npy"), dp_epsilon, radius_range=args.radius_range)
                    col = 'tab:green'
                    sty = 'solid'
                elif name == 'Baseline-DP-size-one':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/dp_baseline_size_one_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'Baseline-Bagging':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/bagging_cpsa.npy"), radius_range=args.radius_range)
                    col = 'tab:purple'
                    sty = 'solid'
                elif name == 'Best-DP':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/best_dp_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'DP-Bagging':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/dp_bagging_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'DP-Bagging-softmax':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/dp_bagging_softmax_cpsa.npy"), radius_range=args.radius_range)

                elif name == r'$\sigma = 1.0$':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_cpsa1.npy"), radius_range=args.radius_range)
                    col = 'tab:orange'
                    sty = 'solid'
                elif name == r'$\sigma = 2.0$':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_cpsa2.npy"), radius_range=args.radius_range)
                    col = 'tab:blue'
                    sty = 'solid'
                elif name == r'$\sigma = 3.0$':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_cpsa3.npy"), radius_range=args.radius_range)
                    col = 'tab:purple'
                    sty = 'solid'
                elif name == r'$\sigma = 4.0$':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_cpsa4.npy"), radius_range=args.radius_range)
                    col = 'tab:green'
                    sty = 'solid'

                elif name == 'ResNet18-500':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/bagging_cpsa_vanilla_500.npy"), radius_range=args.radius_range)
                    col = 'tab:red'
                    sty = 'dashed'
                elif name == 'ResNet18-3000':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/bagging_cpsa_vanilla_3000.npy"), radius_range=args.radius_range)
                    col = 'tab:red'
                    sty = 'solid'
                elif name == 'ResNet18-DP-500':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/bagging_cpsa_500.npy"), radius_range=args.radius_range)
                    col = 'tab:blue'
                    sty = 'dashed'
                elif name == 'ResNet18-DP-3000':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/bagging_cpsa_3000.npy"), radius_range=args.radius_range)
                    col = 'tab:blue'
                    sty = 'solid'
                

                else:
                    print('Invalid method name in Plot.')
                acc_list.append(acc)
                rad_list.append(rad)
                color_list.append(col)
                linestyle_list.append(sty)
            plot_certified_acc(acc_list, rad_list, method_name, color_list, linestyle_list, f"{result_folder}/Bagging_ResNet18_vanilla_vs_DP.png")

            # sub_range = [60000, 30000, 20000]
            # cpsa_dp_list = []
            # cpsa_rdp_list = []
            # for sub in sub_range:
            #     cpsa_dp_list.append(np.load(f"{result_folder}/dp_cpsa_{sub}.npy"))
            #     cpsa_rdp_list.append(np.load(f"{result_folder}/rdp_cpsa_{sub}.npy"))
            
            # acc_rad_dp = [certified_acc_against_radius(cpsa_dp, radius_range=args.radius_range) for cpsa_dp in cpsa_dp_list]
            # acc_rad_rdp = [certified_acc_against_radius(cpsa_rdp, radius_range=args.radius_range) for cpsa_rdp in cpsa_rdp_list]

            # plot_certified_acc([x[0] for x in acc_rad_dp], [x[1] for x in acc_rad_dp], [f'Sub-training size {sub}' for sub in sub_range], f"{result_folder}/compare_certified_acc_plot_sub_dp.png")
            # plot_certified_acc([x[0] for x in acc_rad_rdp], [x[1] for x in acc_rad_rdp], [f'Sub-training size {sub}' for sub in sub_range], f"{result_folder}/compare_certified_acc_plot_sub_rdp.png")


        # plot_certified_acc([[0.930273, 0.876660, 0.809863, 0.590527, 0.353320], [0.916016, 0.856348, 0.774219, 0.480566, 0.179883]], [[50000, 20000, 10000, 3000, 500], [50000, 20000, 10000, 3000, 500]], ['ResNet18', 'ResNet18-DP'], ['tab:red', 'tab:blue'], ['solid', 'solid'], f"{result_folder}/Bagging_ResNet18_vanilla_vs_DP_2.png")
  

        # # Optional "epoch V.S. acc" and "epoch V.S. eps" plots
        # epoch_acc_eps = np.load(f"{result_folder}/epoch_acc_eps.npy")
        # acc_list = [x[0] for x in epoch_acc_eps]
        # eps_list = [x[1] for x in epoch_acc_eps]
        # epoch_list = list(range(1, len(epoch_acc_eps)+1))
        # plot_certified_acc([acc_list], [epoch_list], ['acc'], f"{result_folder}/epoch_vs_acc.png", xlabel='Number of epochs', ylabel='Clean Accuracy')
        # plot_certified_acc([eps_list], [epoch_list], ['eps'], f"{result_folder}/epoch_vs_eps.png", xlabel='Number of epochs', ylabel='DP epsilon')

        # # Optional "sub-training-size V.S. acc" plot
        # acc_lists = []
        # subset_lists = []
        # for sigma in [0.5, 1.0, 2.0]:
        #     subset_acc = np.load(f"{result_folder}/subset_acc_list_{sigma}.npy")
        #     subset_list = [x[0] for x in subset_acc]
        #     acc_list = [x[1] for x in subset_acc]
        #     acc_lists.append(acc_list)
        #     subset_lists.append(subset_list)
        # plot_certified_acc(acc_lists, subset_lists, ['Sigma-0.5', 'Sigma-1.0', 'Sigma-2.0'], f"{result_folder}/subset_vs_acc.png", xlabel='Size of sub-training set', ylabel='Clean Accuracy')

        # subset_acc = np.load(f"{result_folder}/subset_acc_list.npy")
        # subset_list = [x[0] for x in subset_acc]
        # acc_list = [x[1] for x in subset_acc]
        # plot_certified_acc([acc_list], [subset_list], ['Sigma-2.0'], f"{result_folder}/subset_vs_acc.png", xlabel='Size of sub-training set', ylabel='Clean Accuracy')