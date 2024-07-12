### The sources was from https://github.com/shaoxiongji/federated-learning

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid, cifar_iid, prop_noniid, cifar_prop_dirichlet, DBA_testdata
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.test import test_img
from models.Update import LocalUpdate_DBA
from models.Fed import FedAvg
from models.Fed import FedAvg_ver2
from models.Fed import FedAvg_w_Krum
from models.Fed import FedAvg_w_multiKrum
from models.Fed import FedAvg_w_trimmedmean
from models.Fed import FedAvg_w_median
from models.Fed import FedAvg_FoolsGold
#from models.Fed import FedAvg_w_Auror
#from models.Fed import FedAvg_w_AFA
from models.Fed import FedAvg_FLAME

from models.Update import LocalUpdate, LocalUpdate_FullAtt, LocalUpdate_LF_RG, LocalUpdate_CaS

import os, time
import datetime
from models.Nets import ResNet18
from torchtext.datasets import WikiText2



if __name__ == '__main__':
    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and \
                               args.gpu != -1 else 'cpu')

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    
    if args.is_attack == 'true':
        if args.DBA:
            print('\033[36m'+"Attack : "+'\033[0m'+"DBA")
        if args.Full_knowledge_Krum or args.Full_knowledge_median:
            print('\033[36m'+"Attack : "+'\033[0m'+"Full knowledge Attack for Krum or median")
        if args.Random_Gaussian or args.Label_flipping:
            print('\033[36m'+"Attack : "+'\033[0m'+"Random Gaussian or Label flipping")
        if args.Constrain_and_scale:
            print('\033[36m'+"Attack : "+'\033[0m'+"Constain and Scale")
    else:
        print('\033[36m'+"Attack : "+'\033[0m'+"None")
    
    if args.is_defense=='true':
        if args.Krum:
            print('\033[36m'+"Defense : "+'\033[0m'+"Krum")
        if args.FoolsGold: ## later
            print('\033[36m'+"Defense : "+'\033[0m'+"FoolsGold")
        if args.Auror: ## later
            print('\033[36m'+"Defense : "+'\033[0m'+"Auror")
        if args.AFA: ## later
            print('\033[36m'+"Defense : "+'\033[0m'+"AFA")
        if args.median: ## later
            print('\033[36m'+"Defense : "+'\033[0m'+"median")
        if args.FLAME: ## now trying to (22.11.01.)
            print('\033[36m'+"Defense : "+'\033[0m'+"FLAME")
    else:
        print('\033[36m'+"Defense : "+'\033[0m'+"None")
        
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = prop_noniid(dataset_train, args.num_users, args.prob)
        print('\033[36m'+"Dataset : "+'\033[0m'+" mnist")
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                          ])
        trans_test = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = prop_noniid(dataset_train, args.num_users, args.prob)
        print('\033[36m'+"Dataset : "+'\033[0m'+" Cifar10")
    elif args.dataset == 'sentiment':
        trans_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                          ])
        trans_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = prop_noniid(dataset_train, args.num_users, args.prob)
        print('\033[36m'+"Dataset : "+'\033[0m'+" Cifar10")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        #net_glob = CNNCifar(args=args).to(args.device)
        net_glob = ResNet18(name='Local', created_time=current_time).to(args.device)
        #net_glob = ResNet18(name='Local', created_time=current_time)
        print('\033[36m'+"Model : "+'\033[0m'+" ResNet18")
        print('\033[36m'+"batch size : "+'\033[0m' + "{}".format(args.bs))
        print('\033[36m'+"Learning Rate : " +'\033[0m'+ "{}".format(args.lr))
        
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        print('\033[36m'+"Model : "+'\033[0m'+ args.model)
        print('\033[36m'+"batch size : " +'\033[0m'+ "{}".format(args.bs))
        print('\033[36m'+"Learning Rate : "+'\033[0m' + "{}".format(args.lr))
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        print('\033[36m'+"Model : "+'\033[0m'+ args.model)
        print('\033[36m'+"batch size : "+'\033[0m' + "{}".format(args.bs))
        print('\033[36m'+"Learning Rate : "+'\033[0m' + "{}".format(args.lr))
    else:
        exit('Error: unrecognized model')
    #print(net_glob)
    
    m = max(int(args.frac * args.num_users), 1)
    f = int(args.frac * args.num_comps)
    if args.DBA:
        f = int(args.frac * args.DBA_num_comps)
    
    print('\033[36m'+"Number of clients is : "+'\033[0m'+"{}".format(args.num_users))
    print('\033[36m'+"Client number for aggregate is : "+'\033[0m'+"{}".format(m))
    print('\033[36m'+"Attacked client is : "+'\033[0m'+"{}".format(f))
    
    ## Loading the model
    start_iters = 0
    if args.resume_iters:
        start_iters = args.resume_iters
        w_path = os.path.join(args.model_save_dir, "{}-W.ckpt".format(args.resume_iters))
        accu_path = os.path.join(args.model_save_dir, "{}-accu-loss.ckpt".format(args.resume_iters))

        net_glob.load_state_dict(torch.load(w_path))

        tmp = torch.load(accu_path)

        ## Later should modified
        #acc_train_tmp = tmp['main_accu_train']; acc_test_tmp = tmp['main_accu_test'];
        #loss_train_tmp = tmp['loss_train']; loss_test_tmp = tmp['loss_test'];        
        print("\n")
        print("From "+'\033[36m'+"{} rnd".format(start_iters)+'\033[0m'+" start!")

    net_glob_local = copy.deepcopy(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    w_glob_local = net_glob.state_dict()

    # training
    attack_list = []
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    acc_train = []; loss_train = []; acc_test = []; loss_test = []
    if args.Backdoor:
        acc_test_poison = []; loss_test_poison = []

    #if args.resume_iters: ## Later should modified
    #    acc_train.append(acc_train_tmp); acc_test.append(acc_test_tmp);
    #    loss_train.append(loss_train_tmp); loss_test.append(loss_test_tmp);

    

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        loss_locals = [0. for i in range(args.num_users)]
    else:
        w_locals = [w_glob for i in range(m)]
        loss_locals = [0. for i in range(m)]

    if os.path.isfile('./Att_model.npy'):
        os.remove('./Att_model.npy')

    if args.DBA:
        acc_global = []
        acc_local1 = []
        acc_local2 = []
        acc_local3 = []
        acc_local4 = []
        global_set, local1_set, local2_set, local3_set, local4_set = DBA_testdata(dataset_test, args.poison_label)
    
    loss_avg = 0.0
    for iter in range(start_iters, args.epochs):
        
        print('\033[36m'+"{} rnd start!".format(iter)+'\033[0m')
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("idxs_users is : ", idxs_users)

        if args.DBA: ### DAB attack and training
            if args.is_attack=='true':
                trigger_num = 0
        for idx in range(len(idxs_users)):
        ### Benign training
            if args.is_attack=='false':
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idxs_users[idx]])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals[idx] = copy.deepcopy(w)
                loss_locals[idx] = copy.deepcopy(loss)
            else:
                if idx < m - f:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idxs_users[idx]])
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    w_locals[idx] = copy.deepcopy(w)
                    loss_locals[idx] = copy.deepcopy(loss)


                else: ### Malicious training
                    if args.Full_knowledge_Krum or args.Full_knowledge_median:
                        if iter == 0:
                            print("This is full knowledge Attack for Krum or median")
                        local = LocalUpdate_FullAtt(args=args, dataset=dataset_train, idxs=dict_users[idxs_users[idx]], \
                                                idx=idx, total_num=m, w_local=w_locals, iters=iter, \
                                                epoch = args.epochs)
                    elif args.Random_Gaussian or args.Label_flipping:
                        if iter == start_iters:
                            print("This is Random Gaussian or Label flipping")
                        local = LocalUpdate_LF_RG(args=args, dataset=dataset_train, idxs=dict_users[idxs_users[idx]], \
                                            total_num=m)
                    elif args.Backdoor:
                        if iter == start_iters:
                            print("This is Backdoor")

                        if args.DBA:
                            local = LocalUpdate_DBA(args=args, dataset=dataset_train, idxs=dict_users[idxs_users[idx]])
                            w, loss = local.train(epoch=iter, net=copy.deepcopy(net_glob).to(args.device), number = trigger_num)
                            trigger_num += 1
                            w_locals[idx] = copy.deepcopy(w)
                            loss_locals[idx] = copy.deepcopy(loss)

                        if args.Constrain_and_scale: ## We should change as Backdoor later
                            if iter == start_iters:
                                print("This is CaS for backdoor attack")
                            net_locals = copy.deepcopy(w_locals)    
                            local = LocalUpdate_CaS(args=args, dataset=dataset_train, idxs=dict_users[idxs_users[idx]], idx=idx, \
                                                total_num=m, epoch_iter=iter)
                            w, loss, accu_test_poison, loss_test_poison = \
                                local.train(net=copy.deepcopy(net_glob).to(args.device), local_w = net_locals)
                            w_locals[idx] = copy.deepcopy(w)
                            loss_locals[idx] = copy.deepcopy(loss)
                                
                    #######################################################
                    ### We should insert another backdoors in this line ###
                    #######################################################

        # update global weights
        print("Global aggregation start!")
        num_users = m; num_comps = f;
        print("num_users and num_comps :", num_users, num_comps)

        if args.Krum:
            w_glob, losses = FedAvg_w_multiKrum(w_locals, loss_locals, num_users, num_comps)
        elif args.FoolsGold: ## later
            w_glob, losses = FedAvg_FoolsGold(w_locals, w_glob, loss_locals, m, lr = args.lr)
        elif args.Auror: ## later
            w_glob, losses = FedAvg(w_locals, loss_locals)
        elif args.AFA: ## later
            w_glob, losses = FedAvg(w_locals, loss_locals)
        elif args.median: ## later
            w_glob, losses = FedAvg(w_locals, loss_locals)
        elif args.FLAME: ## now trying to (22.11.01.)
            w_glob, losses = FedAvg_FLAME(w_locals, w_glob, loss_locals, num_users, m, num_comps)
        else: ## benign FedAvg
            if args.DBA:
                w_glob, losses = FedAvg_ver2(w_locals, w_glob, loss_locals)
            else:
                w_glob, losses = FedAvg(w_locals, loss_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(losses) / len(losses)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        net_glob.eval()
        with torch.no_grad():
            tmp1, tmp2 = test_img(net_glob, dataset_train, args)
            acc_train.append(tmp1); loss_train.append(tmp2);
            tmp3, tmp4 = test_img(net_glob, dataset_test, args)
            acc_test.append(tmp3); loss_test.append(tmp4);

            print("Training accuracy: {:.2f}".format(tmp1))
            print("Main Task Accuracy (MA): {:.2f}".format(tmp3))
            
            if args.DBA:
                global_trigger, _ = test_img(net_glob, global_set, args)
                local1, _ = test_img(net_glob, local1_set, args)
                local2, _ = test_img(net_glob, local2_set, args)
                local3, _ = test_img(net_glob, local3_set, args)
                local4, _ = test_img(net_glob, local4_set, args)
                acc_global.append(global_trigger)
                acc_local1.append(local1)
                acc_local2.append(local2)
                acc_local3.append(local3)
                acc_local4.append(local4)
                
                print("global Attack Success Rate: "+'\033[32m'+"{:.2f}".format(global_trigger)+'\033[0m')
                print("local1 Attack Success Rate: "+'\033[32m'+"{:.2f}".format(local1)+'\033[0m')
                print("local2 Attack Success Rate: "+'\033[32m'+"{:.2f}".format(local2)+'\033[0m')
                print("local3 Attack Success Rate: "+'\033[32m'+"{:.2f}".format(local3)+'\033[0m')
                print("local4 Attack Success Rate: "+'\033[32m'+"{:.2f}".format(local4)+'\033[0m')
            print()

        """
                # Save model checkpoints
        if (iter + 1) % args.model_save_step == 0:
            w_path = os.path.join(args.model_save_dir, "{}-W.ckpt".format(iter + 1))
            accu_path = os.path.join(args.model_save_dir, "{}-accu-loss.ckpt".format(iter + 1))
            torch.save(net_glob.state_dict(), w_path)
            torch.save({'main_accu_train' : acc_train, 'main_accu_test' : acc_test, \
                        'main_loss_train' : loss_train, 'main_loss_test' : loss_test}, accu_path)
            
            print("Saved model checkpoints into {}...".format(args.model_save_dir))
        """

        # This is for BA
        if args.Backdoor:
            plotstep = 100 ## 100 is default in CaS
            if (iter + 1) % (args.model_save_step + plotstep) == 0:
                accu_poison_path = os.path.join(args.model_save_dir, "{}-accu-loss-poison.ckpt".format(iter + 1))
                torch.save({'backdoor_accu' : acc_test_poison, 'backdoor_loss' : loss_test_poison}, \
                            accu_poison_path)

    if args.DBA:
        #for i in range(len(acc_global)):
        #    if i % 10 == 0:
        #        print("[global accuracy : {:.2f}] ".format(acc_global[i]) + "[local1 accuracy : {:.2f}] ".format(acc_local1[i]) + "[local2 accuracy : {:.2f}] ".format(acc_local2[i]) \
        #        + "[local3 accuracy : {:.2f}] ".format(acc_local3[i]) + "[local4 accuracy : {:.2f}]".format(acc_local4[i]))

        print("total global Attack Success Rate:", end=' ')
        for i in range(0,len(acc_global),2):
            if (acc_global[i] < 75.0) :
                print("{:2.2f}".format(acc_global[i]), end=' ')
            else:
                print('\033[32m'+"{:2.2f}".format(acc_global[i])+'\033[0m', end=' ')
        print()
        print("total local1 Attack Success Rate:", end=' ')
        for i in range(0,len(acc_global),2):
            if (acc_local1[i] < 50.0) :
                print("{:2.2f}".format(acc_local1[i]), end=' ')
            else:
                print('\033[32m'+"{:2.2f}".format(acc_local1[i])+'\033[0m', end=' ')
        print()
        print("total local2 Attack Success Rate:", end=' ')
        for i in range(0,len(acc_global),2):
            if (acc_local2[i] < 50.0) :
                print("{:2.2f}".format(acc_local2[i]), end=' ')
            else:
                print('\033[32m'+"{:2.2f}".format(acc_local2[i])+'\033[0m', end=' ')
        print()
        print("total local3 Attack Success Rate:", end=' ')
        for i in range(0,len(acc_global),2):
            if (acc_local3[i] < 50.0) :
                print("{:2.2f}".format(acc_local3[i]), end=' ')
            else:
                print('\033[32m'+"{:2.2f}".format(acc_local3[i])+'\033[0m', end=' ')
        print()
        print("total local4 Attack Success Rate:", end=' ')
        for i in range(0,len(acc_global),2):
            if (acc_local4[i] < 50.0) :
                print("{:2.2f}".format(acc_local4[i]), end=' ')
            else:
                print('\033[32m'+"{:2.2f}".format(acc_local4[i])+'\033[0m', end=' ')
        print()

        if (iter + 1) % 50 == 0:
            plt.figure()
            plt.plot(range(len(acc_global)), acc_global, label = 'global_trigger')
            plt.plot(range(len(acc_local1)), acc_local1, label = 'local1_trigger', marker = 'o', ls = '--')
            plt.plot(range(len(acc_local2)), acc_local2, label = 'local2_trigger', marker = 's', ls = '--')
            plt.plot(range(len(acc_local3)), acc_local3, label = 'local3_trigger', marker = 'x', ls = '--')
            plt.plot(range(len(acc_local4)), acc_local4, label = 'local4_trigger', marker = '^', ls = '--')
            plt.ylabel('Attack Success Rate')
            plt.legend()
            plt.savefig('./save/fed_DBA_FLAME_{}_trigger.png'.format(iter))
        

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, \
                                                           args.frac, args.iid))
