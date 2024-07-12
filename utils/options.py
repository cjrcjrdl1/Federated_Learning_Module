#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    ## federated arguments ##
    parser.add_argument('--epochs', type=int, default=10100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--is_attack', type=str, default='false', help="is attack applied?")
    parser.add_argument('--is_defense', type=str, default='false', help="is defense applied?")

    parser.add_argument('--num_comps', type=int, default=10, help="number of compromised users: f")
    parser.add_argument('--prob', type=float, default=0.1, help="probability of non-iidness")

    # This is not used for original FL systems. It is for Blended FL by ours.
    #parser.add_argument('--is_Blended', action='store_true', help="is this system for Blended FL?")
    #parser.add_argument('--central_bs', type=int, default=1000, help="batch for central DB: B2")

    ## model arguments ##
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    ## other arguments ##
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_false', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--model_save_step', type=int, default=5000, help='model save step')
    parser.add_argument('--model_save_dir', type=str, default="savemodels", help='model save dir')
    parser.add_argument('--resume_iters', type=int, default=10000, help='resume iters')

    ### There are only one 'store_false' in attacks and defences each ###
    ## For attacks
    parser.add_argument('--Full_knowledge_Krum', action='store_true', help='Full knowledge for Krum')
    parser.add_argument('--Full_knowledge_median', action='store_true', help='Full knowledge for median')
    parser.add_argument('--Random_Gaussian', action='store_true', help='Random Gaussian attack')
    parser.add_argument('--Label_flipping', action='store_true', help='Label flipping')

    ## For attacks - backdoors
    parser.add_argument('--Backdoor', action='store_true', help='Backdoored is activated')
    # Kinds of backdoors 
    parser.add_argument('--Constrain_and_scale', action='store_true', help='attacks[7] in FLAME')
    parser.add_argument('--DBA', action='store_true', help='attacks[59] in FLAME')
    parser.add_argument('--Edge_Case', action='store_true', help='attacks[57] in FLAME')
    parser.add_argument('--PGD', action='store_true', help='attacks[57] in FLAME')

    ## For defenses
    parser.add_argument('--Krum', action='store_true', help='defenses using Krum')
    parser.add_argument('--FoolsGold', action='store_true', help='defenses using FoolsGold')
    parser.add_argument('--Auror', action='store_true', help='defenses using Auror')
    parser.add_argument('--AFA', action='store_true', help='defenses using AFA')
    parser.add_argument('--median', action='store_true', help='defenses using median')
    parser.add_argument('--FLAME', action='store_true', help='defenses using FLAME')

    ########## Attack line ##########
    ### (maybe) default
    parser.add_argument('--save_on_epochs', type=int, default=1000, help='saving point in CaS, must be smaller than epoch. default is 1000')
    parser.add_argument('--attpoint', type=int, default=10002, help='attack point for BA. default is 995')

    ### for DBA ###############
    parser.add_argument('--global_trigger', type=bool, default=True, help='global_trigger in DBA')
    parser.add_argument('--DBA_num_comps', type=int, default=40, help="DBA_number of compromised users: f")
    parser.add_argument('--shot', action='store_true', help="DBA shot style")
    parser.add_argument('--poison_label', type=int, default=2, help="DBA change label")
    parser.add_argument('--poison_count', type=int, default=7, help="DBA trigger image num")
    parser.add_argument('--DBA_lr', type=float, default=0.05, help="poison lr in DBA")
    parser.add_argument('--glob', action='store_true', help="DBA trigger style")
    parser.add_argument('--DBA_ep', type=int, default=8, help="the number of poison epochs: E")
    parser.add_argument('--Alpha_loss', type=float, default=1.0, help='alpha_loss in DBA')
    
    ### For CaS : please compare between this file and utils/params.yaml in CaS
    parser.add_argument('--test_batch_size', type=int, default=1000, help='Test batch size for CaS')
    parser.add_argument('--decay', type=float, default=0.0005, help='decay defined in CaS')
    parser.add_argument('--no_models', type=int, default=10, help='# of models in CaS')
    parser.add_argument('--retrain_no_times', type=int, default=6, help='retrain times in CaS')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.9, help='dirichlet probability for data distributions in CaS')
    parser.add_argument('--eta', type=int, default=1, help='eta in CaS')
    parser.add_argument('--resumed_model', action='store_true', help='resume model in CaS')
    parser.add_argument('--poison_type', type=str, default='wall', help='poison type in CaS')
    parser.add_argument('--random_compromise', action='store_true', help='random compromise in CaS')
    parser.add_argument('--noise_level', type=float, default='0.01', help='noise level in CaS')
    parser.add_argument('--poison_epochs', type=int, default=10005, help='poison epochs in CaS')
    parser.add_argument('--poison_lr', type=float, default=0.05, help='poison lr in CaS')
    parser.add_argument('--poison_bs', type=int, default=20, help='poison batch in CaS written by me')
    parser.add_argument('--poison_momentum', type=float, default=0.9, help='poison momentum in CaS')
    parser.add_argument('--poison_decay', type=float, default=0.005, help='poison decay in CaS')
    parser.add_argument('--poison_step_lr', action='store_false', help='poison_step_lr in CaS')
    parser.add_argument('--clamp_value', type=float, default=1.0, help='clamp_value in CaS')
    parser.add_argument('--alpha_loss', type=float, default=1.0, help='alpha_loss in CaS')
    parser.add_argument('--number_of_adversaries', type=int, default=1, help='# of advs in CaS')
    parser.add_argument('--poisoned_number', type=int, default=2, help='poisoned number in CaS')
    parser.add_argument('--s_norm', type=int, default=1000000, help='s_norm in CaS')
    parser.add_argument('--baseline', action='store_false', help='if not, the poison_lr is decayed')

    ## Is DP used or not? It will be added later (e.g., DP2 by Vincent Poor et al.)
    parser.add_argument('--DP', action='store_false', help='Is DP used or not?')

    ## For ERR-based
    parser.add_argument('--ERR', action='store_false', help='whether ERR-based used or not')

    ## For DP
    parser.add_argument('--sigma', type=float, default=14., help='sigma using epsilon and delta in DL with DP paper(by Google)')
    parser.add_argument('--clipping', type=float, default=4., help='clipping threshold')

    args = parser.parse_args()
    return args
