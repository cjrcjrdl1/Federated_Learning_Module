#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import math, os, time
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

''' 

IMPORTANT NOTES) We have to insert 
                 (1) Constrain_and_scale
                 (2) DBA
                 (3) Edge_Case
                 (4) PGD
            irint(net_dict)
            print(net_dict)

Attack method

### #1. def Random_Gaussian
### #2. def Full_for_median
### #3. def Full_for_Krum

class
*** NOTE) Random_Gaussian and Label_flipping are in LocalUpdate class ***
*** NOTE) There is benign LocalUpdate in LocalUpdate_DP1 and LocalUpdate_DP2 //
          Therefore, we can just comment for adding noise phase

### #1. class LocalUpdate_FullAtt(object):      ## LocalUpdate with FullAtt
### #2. class LocalUpdate_LF_RG(object):        ## LocalUpdate with LF, RG
### #4. class LocalUpdate_CAS(object):          ## LocalUpdate with Constrain-and-Scale att
### #5. class LocalUpdate_DBA(object):          ## LocalUpdate with DBA att
### #6. class LocalUpdate_Edgecase(object):     ## LocalUpdate with Edgecase
### #7. class LocalUpdate_PGD(object):          ## LocalUpdate with PGD
### #8. class LocalUpdate(object):              ## This is used for benign users such as in FullAtt

### for DP1 : Later
### for DP2 : Later if we need ...

'''
########## Attack method ##########

#### Attack method 1 : Random Gaussian
#### This method is defined in Krum paper(NIPS 2017)
def Random_Gaussian(net_dict):
    nets = copy.deepcopy(net_dict)
    for key in nets.keys():
        if nets[key].dim() == 0:
            #nets[key] = torch.randn(1) * std + mu
            nets[key] = torch.randn(1).normal_(mean=0., std=200.**2)
        else:
            nets[key] = torch.randn_like(nets[key]).normal_(mean=0., std=200.**2)
    return nets

#### Attack method 2 : Full knowledge (model poisoning)
#### For trimmed mean, median
######################### Should be modified later because of fraction
def Full_for_median(global_net, net_dict, w_local, total_num, num_of_comp, iters, args):

    if os.path.isfile('./median_attmodel.npy'):
        net_tmp = np.load('./median_attmodel.npy', allow_pickle=True)
        net_tmp = net_tmp.item()
        if os.path.isfile('./direc.npy'):
            b = np.load('./direc.npy', allow_pickle=True)
            direc = b.item()
        return net_tmp, direc
    else:
        # Set # of workers
        m = total_num; c = num_of_comp;

        net_tmp = copy.deepcopy(net_dict)
        max_avg = copy.deepcopy(w_local[0]); min_avg = copy.deepcopy(w_local[0]);

        drts = copy.deepcopy(global_net)
        drts_tmp = copy.deepcopy(global_net)

        #net_tmp = torch.zeros_like(net_tmp)

        ### compute direction "s" using before global weight and current global weight
        ### current global weight is computed by you.
        mag_check = 0; s_j = 0;
        net_glob_tmp = copy.deepcopy(net_dict); ## global tmp
        w_avg = copy.deepcopy(net_dict);        ## aggregated local tmp

        w_avg = AttAvg_w_median(w_local, args.num_users, m, args.num_comps)

        for k in net_glob_tmp.keys():
            #net_glob_tmp[k] = w_avg[k] - global_net[k]
            net_glob_tmp[k] = w_avg[k]
            mag_check += np.sum(net_glob_tmp[k].cpu().numpy())
            #norm_check += np.linalg.norm(net_glob_tmp[k].cpu().numpy())
        if (mag_check > 0. or mag_check == 0.):
            s_j = 1
        else:
            s_j = -1

        for k in max_avg.keys():
            max_avg[k] = torch.zeros_like(max_avg[k])
            min_avg[k] = torch.zeros_like(min_avg[k])

        '''
        ### compute direction "s" using before global weight and current local weight
        ### This is different from the original paper, because computation in this paper
        ### is weird.

        norm_check = 0; s_j = 0;
        for k in global_net.keys():
            drts[k] = net_dict[k] - global_net[k]
            norm_check += np.linalg.norm(drts[k].cpu().numpy())

        if (norm_check > 0.):
            s_j = 1
        else:
            s_j = -1

        for k in max_avg.keys():
            max_avg[k] = torch.zeros_like(max_avg[k])
            min_avg[k] = torch.zeros_like(min_avg[k])
        '''

        ### get max and min local weight on benign weights
        check_list = np.array([0.0 for j in range(len(w_local))], dtype=np.float128)
        for i in range(len(w_local)):
            for k in max_avg.keys():
                check_list[i] += np.sum(w_local[i][k].cpu().numpy())
                #check_list[i] += np.linalg.norm(w_local[i][k].cpu().numpy())

        check_list_tmp = np.sort(check_list)
        check_arglist = np.argsort(check_list)

        max_idx = check_arglist[-1]; min_idx = check_arglist[0];

        #print("Check list :", check_list, max_idx, min_idx)
        #print(check_arglist)
        #print("maxidx, minidx ", max_idx, min_idx)

        max_avg = copy.deepcopy(w_local[max_idx])
        min_avg = copy.deepcopy(w_local[min_idx])

        ### compute compromised weight using s_j
        ## compute max_avg and min_avg is bigger than zero
        max_norm = 0.; min_norm = 0.; const_b = 2;
        for k in max_avg.keys():
            max_norm += np.sum(max_avg[k].cpu().numpy(), dtype=np.float128)
            min_norm += np.sum(min_avg[k].cpu().numpy(), dtype=np.float128)
            #max_norm += np.linalg.norm(max_avg[k].cpu().numpy())
            #min_norm += np.linalg.norm(min_avg[k].cpu().numpy())
        print(max_norm, type(max_norm))

        print("mag, s_j,  max_norm, min_norm is :", mag_check, s_j, max_norm, min_norm)

        if s_j == -1 :
            if max_norm > 0.:
                #print("case 0")
                direc = 1
                #b = np.random.uniform(1, const_b, size=1)
                for k in global_net.keys():
                    #b = torch.rand_like(max_avg[k]) * const_b
                    net_tmp[k] = max_avg[k]
                    #net_tmp[k] = b * max_avg[k] 
            else:
                #print("case 1")
                direc = 2
                #b = np.random.uniform(1/const_b, 1, size=1)
                for k in global_net.keys():
                    #b = torch.rand_like(max_avg[k]) * (1 / const_b)
                    net_tmp[k] = max_avg[k]
                    #net_tmp[k] = b * max_avg[k]
        elif s_j == 1 :
            if min_norm > 0.:
                #print("case 2")
                direc = 3
                #b = np.random.uniform(1/const_b, 1, size=1)
                for k in global_net.keys():
                    #b = torch.rand_like(max_avg[k]) * (1 / const_b)
                    net_tmp[k] = min_avg[k]
                    #net_tmp[k] = b * max_avg[k]
            else:
                #print("case 3")
                direc = 4
                #b = np.random.uniform(1, const_b, size=1)
                for k in global_net.keys():
                    #b = torch.rand_like(max_avg[k]) * const_b
                    net_tmp[k] = min_avg[k]
                    #net_tmp[k] = b * max_avg[k]

        np.save('./median_attmodel', net_tmp)
        np.save('./direc', direc)

    return net_tmp, direc

###### This works only for the one of attackers
#### Attack method 2 : Full knowledge (model poisoning)
#### I think, this method attacks Krum using some techniques in the multi-Krum
#### Especially, find lambda
def Full_for_Krum(global_net, net_dict, w_local, total_num, num_of_comp, iters, args):

    if os.path.isfile('./krum_attmodel.npy'):
        net_tmp = np.load('./krum_attmodel.npy', allow_pickle=True)
        net_tmp = net_tmp.item()
    else:
        net_tmp = copy.deepcopy(net_dict)

        ### w_local = m - f
        score = np.array([0.0 for i in range(len(w_local))], dtype=np.float128)

        w_avg = copy.deepcopy(w_local[0])

        for k in w_avg.keys():
            w_avg[k] = torch.zeros_like(w_avg[k])

        tmp = copy.deepcopy(w_local)

        for i in range(len(w_local)): ## for each local models
            w_tmp = 0
            for j in range(len(w_local)):
                if j == i :
                    continue
                else:
                    sum_tmp = 0
                    for k in w_avg.keys():
                        A = tmp[i][k].cpu().numpy(); A = np.array(A, dtype=np.float128)
                        B = tmp[j][k].cpu().numpy(); B = np.array(B, dtype=np.float128)
                        arr_tmp = A - B
                        arr_tmp = np.power(arr_tmp, 2)
                        sum_tmp += np.sum(arr_tmp)
                    score[i] += sum_tmp
            if i == 1:
                print("score[i] in Full_for_Krum :", score[i], type(score[i]))
            if np.isnan(score[i]) == True:
                print("End in Full_for_Krum")
                quit()

        '''
        for k in w_avg.keys(): ## for each key
            for i in range(len(w_local)): ## for each local models
                w_tmp = 0
                for j in range(len(w_local)):
                    if j == i:
                        continue
                    else:
                        A = tmp[i][k].cpu().numpy()
                        B = tmp[j][k].cpu().numpy()
                        #w_tmp = np.linalg.norm(A - B, ord=2)
                        w_tmp = np.linalg.norm(A - B, ord=None)
                        w_tmp = w_tmp ** 2
                score[i] += w_tmp
        '''

        ## compute dim
        num_param = 0
        for k in w_avg.keys(): ## for each key
            tmp = w_avg[k]
            Dim = torch.Tensor.dim(tmp)

            if Dim == 3:
                a, b, c = tmp.shape
                total = a * b * c
                num_param += total
            elif Dim == 2:
                a, b = tmp.shape
                total = a * b
                num_param += total
            else:
                a = tmp.shape
                total = list(a)
                for i in range(len(total)):
                    num_param += total[i]

        tmp = copy.deepcopy(w_local)

        ### Find lambda
        m = total_num; c = num_of_comp

        const = (m - 2*c -1) * math.sqrt(num_param)
        const = 1 / const

        ### Compute min and value
        sorted_idx_score = np.argsort(score)
        idx = m - c
        min_value = 0.0
        max_value= 0.0
        min_list = np.array([0.0 for i in range(idx)], dtype=np.float128)
        max_list = np.array([0.0 for i in range(idx)], dtype=np.float128)

        for i in range(idx): ## for each local model
            ### Get a min value ###
            for j in range(idx - 2):
                if sorted_idx_score[j] == i:
                    continue
                else:
                    sum_tmp = 0
                    for k in w_avg.keys():
                        A = tmp[i][k].cpu().numpy()
                        A = np.array(A, dtype=np.float128)
                        B = tmp[sorted_idx_score[j]][k].cpu().numpy()
                        B = np.array(B, dtype=np.float128)
                        arr_tmp = A - B
                        arr_tmp = np.power(arr_tmp, 2)
                        sum_tmp += np.sum(arr_tmp)
                    min_list[i] += sum_tmp
            #######################

            ### Get a max value ###
            sum_tmp2 = 0
            for k in w_avg.keys():
                A = tmp[i][k].cpu().numpy()
                A = np.array(A, dtype=np.float128)
                B = tmp[j][k].cpu().numpy()
                B = np.array(B, dtype=np.float128)
                arr_tmp2 = A - B
                arr_tmp2 = np.power(arr_tmp2, 2)
                sum_tmp2 += np.sum(arr_tmp2)
            max_list[i] += sum_tmp2
            #######################

        '''
        for k in w_avg.keys(): ## for each key
            for i in range(idx):
                ### Get a min value ###
                w_tmp = 0
                for j in range(idx - 2):
                    if sorted_idx_score[j] == i:
                        continue
                    else:
                        A = tmp[i][k].cpu().numpy()
                        B = tmp[sorted_idx_score[j]][k].cpu().numpy()
                        w_tmp = np.linalg.norm(A - B, ord=None)
                        w_tmp = w_tmp ** 2
                min_list[i] += w_tmp
                #######################
    
                ### Get a max value ###
                w_tmp = 0
                A = tmp[i][k].cpu().numpy()
                B = global_net[k].cpu().numpy()
                w_tmp = np.linalg.norm(A - B, ord=None)
                w_tmp = w_tmp ** 2
    
                max_list[i] += w_tmp
                #######################
        '''

        min_list = np.sort(min_list)
        min_value = min_list[0]

        max_list = np.sort(max_list)
        max_value = max_list[len(max_list) - 1]

        fst_eq = const * min_value
        snd_eq = 1 / math.sqrt(num_param) * max_value

        ### results for Upper threshold
        upper_th = fst_eq + snd_eq

        ### results for Lower threshold
        lower_th = (1 / (10 ** 2))

        #if iters == 0:
            #print("Th are : ", upper_th, lower_th, iters)

        lamb = upper_th
        drts = copy.deepcopy(global_net)
        drts_tmp = copy.deepcopy(global_net)

        if iters == 0:
            ### To get direction vector, "s"
            np.save('./before_global', global_net)

            ### To get lambda to compare whether comp value = next global or not
            np.save('./before_local', net_tmp)
            np.save('./lamb', lamb)

            ### compute direction "s" using zero weight for 0th training
            pre_w = copy.deepcopy(global_net)
            for k in pre_w.keys():
                pre_w[k] = torch.zeros_like(global_net[k])

            for k in global_net.keys():
                drts[k] = global_net[k] - pre_w[k]
                drts_tmp[k] = torch.where(drts[k] >= 0., 1., -1.)

        else:
            ### To get "s"
            pre_w = np.load('./before_global.npy',allow_pickle=True); pre_w = pre_w.item()

            ## compute direction "s"
            for k in pre_w.keys():
                drts[k] = global_net[k] - pre_w[k]
                drts_tmp[k] = torch.where(drts[k] >= 0., 1., -1.)

            np.save('./before_global', global_net)

            ### To get lambda
            net_tmp = np.load('./before_local.npy', allow_pickle=True); net_tmp = net_tmp.item()
            lamb = np.load('./lamb.npy', allow_pickle=True); lamb = lamb.item()

            ### To get comparing lists
            w1 = np.load('./krum_attmodel2.npy', allow_pickle=True); w1 = w1.item()

            boolean = 1
            ''' #### It is not useless in multi-Krum. 
            for k in global_net.keys():
                #if torch.equal(net_tmp[k], global_net[k]) == True:
                if torch.equal(drts[k], w1[k]) == True:
                    boolean = 0
                else:
                    boolean += 1
            '''

            if boolean == 0 or lamb < lower_th:
                print("1 break test")
                lamb = lamb
            else:
                lamb = lamb / 2
                print("2 break test")

            print("lamb is :", lamb)
            np.save('./lamb', lamb)

        weight_tmp = copy.deepcopy(net_tmp)

        ### compute compromised local weight
        for k in global_net.keys():
            #net_tmp[k] = global_net[k]
            #net_tmp[k] = 0. - (lamb * drts_tmp[k]) ## this is only weight except global_net[k]
            #weight_tmp[k] = lamb * drts_tmp[k]
            #net_tmp[k] = global_net[k] - weight_tmp[k]
            net_tmp[k] = lamb * drts_tmp[k]
            #net_tmp[k] = global_net[k] - (lamb * drts_tmp[k])

        np.save('./krum_attmodel', net_tmp)
        np.save('./krum_attmodel2', net_tmp)

    return net_tmp

###################################################################################################
###################################################################################################
###################################################################################################
#####                                         Borderline                                      #####
###################################################################################################
###################################################################################################
###################################################################################################

########## Class ##########

#### No. 1 ####
class LocalUpdate_FullAtt(object):
    def __init__(self, args, dataset=None, idxs=None, idx=None, total_num=None, \
                 w_local=None, iters=None, epoch=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.idx = idx
        self.total_num = total_num
        self.num_of_comp = args.num_comps # the number of compromised workers
        #self.num_of_comp = 10

        #self.w_local = copy.deepcopy(w_local)
        self.w_local = w_local
        self.iters = iters
        self.epoch = epoch

    def train(self, net):
        global_net = copy.deepcopy(net.state_dict()) ## this is previous global weight before computing local value

        net.train()

        net_original = copy.deepcopy(net.state_dict())
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        #print(self.idx)

        ### Malicious training for master
        if self.idx > (self.total_num - self.num_of_comp - 1):
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            ## current net weights
            net_dict = net.state_dict()

            if self.args.Full_knowledge_Krum :
                net_dict_tmp = Full_for_Krum(global_net, net_dict, self.w_local, \
                                         self.total_num, self.num_of_comp, self.iters, self.args)
            else: ## This is for args.Full_knowledge_median
                net_dict_tmp, direc = Full_for_median(global_net, net_dict, self.w_local, \
                                         self.total_num, self.num_of_comp, self.iters, self.args)
                const_b = 2
                if direc == 1:
                    b = np.random.uniform(1, const_b, size=1)
                    for k in global_net.keys():
                        net_dict_tmp[k] = b[0] * net_dict_tmp[k]
                elif direc == 2:
                    b = np.random.uniform(1/const_b, 1, size=1)
                    for k in global_net.keys():
                        net_dict_tmp[k] = b[0] * net_dict_tmp[k]
                elif direc == 3:
                    b = np.random.uniform(1/const_b, 1, size=1)
                    for k in global_net.keys():
                        net_dict_tmp[k] = b[0] * net_dict_tmp[k]
                else:
                    b = np.random.uniform(1, const_b, size=1)
                    for k in global_net.keys():
                        net_dict_tmp[k] = b[0] * net_dict_tmp[k]

            for k in net_dict.keys():
                net_dict[k] = net_dict[k] - net_dict_tmp[k]

        ### benign training
        else:
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            net_dict = copy.deepcopy(net.state_dict())

        return net_dict, sum(epoch_loss) / len(epoch_loss)

#### No. 2 ####
class LocalUpdate_LF_RG(object):
    def __init__(self, args, dataset=None, idxs=None, idx=None, total_num=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.idx = idx
        self.total_num = total_num
        self.num_of_comp = args.num_comps # the number of compromised workers
        #self.num_of_comp = 10

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        if self.args.Label_flipping:
            #print("Label flipping is selected", self.idx)
            L = 10
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    for i in range(len(labels)):
                        labels[i] = L - labels[i] - 1
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            net_dict = copy.deepcopy(net.state_dict())

        elif self.args.Random_Gaussian:
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            ## current net weights
            net_dict = copy.deepcopy(net.state_dict())
            net_dict = Random_Gaussian(net_dict)
        
        return net_dict, sum(epoch_loss) / len(epoch_loss)

#### No. 8 ####
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, \
                                    num_workers=2, persistent_workers=True)
        

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                #print("label is :", labels, len(labels))
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))



from models import test
from models.Function import dist_norm

#### No. 3 ####
class LocalUpdate_CaS(object):
    def __init__(self, args, dataset=None, idxs=None, idx=None, total_num=None, epoch_iter=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.idx = idx
        self.total_num = total_num
        self.num_of_comp = args.num_comps # the number of compromised workers
        #self.num_of_comp = 10
        self.epoch_iter = epoch_iter

        ## To load poison data for train
        # Vertical stripes on background wall
        self.poison_images = [30696, 33105, 33615, 33907, 36848, 40713, 41706]
        ## This is for test by Adv. So this is in trainingset, not testset
        self.poison_images_test = [330, 568, 3934, 12336, 30560]
        self.ldr_train_poison = DataLoader(DatasetSplit(dataset, self.poison_images), \
                                           batch_size=len(self.poison_images), shuffle=True)
        self.ldr_test_poison = DataLoader(DatasetSplit(dataset, self.poison_images_test), \
                                           batch_size=len(self.poison_images_test), shuffle=True)
                                           
    


    def train(self, net, local_w):
        net_test = copy.deepcopy(net) # for testing and call the original grad G^t
        net.train()

        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.poison_lr, momentum=self.args.momentum)

        epoch_loss = []
        att_times = 1
        if self.epoch_iter < self.args.attpoint or self.epoch_iter >= (self.args.attpoint + att_times) :
            net_test.eval()
            with torch.no_grad():
                acc_test_poison, loss_test_poison = \
                        test.test_img_backdoor(net_test, self.ldr_test_poison, self.args)
                #print("Accu for poisoned in deactivated phase : {:.2f}".format(acc_test_poison))

            # train and update
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    global log_probs
                    log_probs = net(images)
                    #global loss_class
                    #print("label is :", labels, len(labels))
                    loss_class = self.loss_func(log_probs, labels)
                    #loss = alpha * loss_class + (1 - alpha) * loss_distance
                    loss_class.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss_class.item()))
                    batch_loss.append(loss_class.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            net_dict = net.state_dict()
       
            
        else:
            ### Testing phase : Testing should be first because of adjusting learning rate ###
            ### line 15 ~ 18
            ### line around 96 in github
            print("start!!")
            net_test.eval()
            with torch.no_grad():
                acc_test_poison, loss_test_poison = \
                        test.test_img_backdoor(net_test, self.ldr_test_poison, self.args)
                #print("Accu for poisoned in activated phase : {:.2f}".format(acc_test_poison))

            for iter in range(self.args.retrain_no_times):
                batch_loss = []

                # train and update using modified poison_lr
                if not self.args.baseline:
                    if acc_test_poison > 20:
                        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.poison_lr/50, \
                                                    momentum=self.args.momentum)
                    elif acc_test_poison > 60:
                        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.poison_lr/100, \
                                                    momentum=self.args.momentum)
                    else:
                        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.poison_lr, \
                                                    momentum=self.args.momentum)
                else:
                    optimizer = torch.optim.SGD(net.parameters(), lr=self.args.poison_lr, \
                                                momentum=self.args.momentum)
                ### Testing phase end ###
                    
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    #images, labels = images.to(self.args.device), labels.to(self.args.device)
                    ## we need to inject c backdoors to the batch b
                    ## example for (c, b) : (20, 64)

                    ## To inject poisoned trainingset
                    for batch_idx_poison, (images_poison, labels_poison) in enumerate(self.ldr_train_poison):
                        images_poison, labels_poison = images_poison, labels_poison
                        for i in range(len(labels_poison)):
                            labels_poison[i] = 2

                    # Line 12 : replace(c, b, D_backdoor)
                    for i in range(self.args.poison_bs): ## ex) until 20
                        #print(i, len(self.poison_images), len(images_poison))
                        images[i] = images_poison[i % len(self.poison_images)]
                        labels[i] = labels_poison[i % len(self.poison_images)]
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                    # Line 13 : compute the gradient
                    net.zero_grad()
                    log_probs1 = net(images)
                    #print("label is :", labels, len(labels))
                    loss_class = self.loss_func(log_probs1, labels)
                    loss_class.backward()
                    optimizer.step()
                    
                    log_probs1 = net(images)
                    #print("label is :", labels, len(labels))
                    loss_class = self.loss_func(log_probs1, labels)
                    loss_distance = dist_norm(net_test.state_dict(), net.state_dict()) 
                    alpha = self.args.alpha_loss
                    loss = alpha * loss_class + (1 - alpha) * loss_distance #
                    
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            ### line 21 : model replacement ###
            # compute gamma
            gamma = self.args.num_users * self.args.frac #m 
            #gamma = self.args.num_users * self.args.frac
            # current net weights : X
            net_dict = net.state_dict()
            # before global weight : G^t
            net_original = net_test.state_dict()

            for k in net_dict.keys():
                net_dict[k] = gamma * (net_dict[k] - net_original[k]) + net_original[k]
            ### Model replacement end ###

        return net_dict, sum(epoch_loss) / len(epoch_loss), acc_test_poison, loss_test_poison

#### No. 4 ####

class LocalUpdate_DBA(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, \
                                    num_workers=2, persistent_workers=True)
    
    def get_poison_batch(self, bptt, poison_patterns, evaluation=False):
        images, targets = bptt

        new_images=images
        new_targets=targets

        attack = True

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing

                new_targets[index] = self.args.poison_label
                new_images[index] = self.add_pixel_pattern(images[index],poison_patterns)
                
            else: # poison part of data when training
                if index < self.args.poison_count:
                #if index < 30:
                    new_targets[index] = self.args.poison_label
                    new_images[index] = self.add_pixel_pattern(images[index],poison_patterns)
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]
                    attack = False

            #new_images = new_images.to(self.args.device)
            #new_targets = new_targets.to(self.args.device)

        return new_images,new_targets, attack

    def add_pixel_pattern(self, ori_image, poison_patterns):
        image = copy.deepcopy(ori_image)

        for i in range(0,len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1  # red : 1, 0.5 0
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1
        #save_image(image,'./trigger_image.png')

        return image

    def train(self, epoch, net, number):
        poison_pattern = [[[0, 1], [0, 2], [0, 3], [0, 4]], [[0, 10], [0, 11], [0, 12], [0, 13]], [[4, 1], [4, 2], [4, 3], [4, 4]], [[4, 10], [4, 11], [4, 12], [4, 13]]]
        net_test = copy.deepcopy(net)
        net_poison = copy.deepcopy(net)
        net_poison.train()
        net.train()

        poison_loss = {}
        loss = {}

        poison_train = True

        attack_num = 4
        # train and update
        adversarial_index = poison_pattern[number]
        attack_epoch = [10504, 10506, 10508, 10510]

        if self.args.glob:
            adversarial_index = []
            attack_epoch = [10510]
            attack_num = 1
            for i in range(len(poison_pattern)):
                adversarial_index = adversarial_index + poison_pattern[i]

        if self.args.shot: ############## single shot ##############
            attack_try = False
            for idx, epo in enumerate(attack_epoch):
                if epoch == epo and number == idx:
                #if epoch == epo:
                    print("This is DBA_single_shot for backdoor attack")
                    optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
                    poison_optimizer = torch.optim.SGD(net_poison.parameters(), lr=self.args.DBA_lr, momentum=self.args.momentum, weight_decay=0.0005)
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * 7,
                                                                             0.8 * 7], gamma=0.1)
                    attack_try = True
                    self.args.poison_count = 30
                    epoch_loss = []
                    print(adversarial_index)
                    for iter in range(self.args.DBA_ep):
                        last_local_model = dict()
                        #local_model_update_dict = dict()
                        for name, data in net.state_dict().items():
                            last_local_model[name] = net.state_dict()[name].clone()
                        batch_loss = []
                        batch_poison_loss = []
                        for batch_idx, (batch) in enumerate(self.ldr_train):
                            images, labels, poison_train = self.get_poison_batch(batch, poison_patterns=adversarial_index,evaluation=True)
                            images, labels = images.to(self.args.device), labels.to(self.args.device)


                            net_poison.zero_grad()
                            output = net_poison(images)
                            poison_loss = self.loss_func(output, labels)
                            poison_loss.backward()
                            poison_optimizer.step()
                            batch_poison_loss.append(poison_loss.item())

                        epoch_loss.append((sum(batch_loss)+sum(batch_poison_loss))/(len(batch_loss)+len(batch_poison_loss)))
                        scheduler.step()

                        for key, value in net_poison.state_dict().items():
                            target_value  = net_test.state_dict()[key]
                            new_value = target_value + (value - target_value) * 100
                            net.state_dict()[key].copy_(new_value)
                    


            if attack_try == False:
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

                epoch_loss = []
                for iter in range(self.args.local_ep):
                    batch_loss = []
                    for batch_idx, (images, labels) in enumerate(self.ldr_train):
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        net.zero_grad()
                        log_probs = net(images)
                        #print("label is :", labels, len(labels))
                        loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        optimizer.step()

                        batch_loss.append(loss.item())
                    epoch_loss.append(sum(batch_loss)/len(batch_loss))
        else: ############## multi shot ##############
            if epoch < 10700 and number < attack_num:
                print("This is DBA_multi_shot for backdoor attack")
                poison_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.DBA_lr, momentum=self.args.momentum, weight_decay=0.0005)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                milestones=[0.2 * 6,
                                                                            0.8 * 6], gamma=0.1)
                epoch_loss = []
                for iter in range(self.args.DBA_ep):
                    batch_loss = []
                    for batch_idx, (batch) in enumerate(self.ldr_train):
                        images, labels, poison_train = self.get_poison_batch(batch, poison_patterns=adversarial_index,evaluation=False)
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        net.zero_grad()
                        log_probs = net(images)
                        #print("label is :", labels, len(labels))
                        class_loss = self.loss_func(log_probs, labels)
                        distance_loss = dist_norm(net_test.state_dict(), net.state_dict())
                        alpha = self.args.Alpha_loss
                        loss = alpha * class_loss + (1 - alpha) * distance_loss
                        #loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        poison_optimizer.step()
                        """
                        if self.args.verbose and batch_idx % 10 == 0:
                            print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                    100. * batch_idx / len(self.ldr_train), loss.item()))
                        """
                        batch_loss.append(loss.item())
                    epoch_loss.append(sum(batch_loss)/len(batch_loss))
                    scheduler.step()
                

            else:
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

                epoch_loss = []
                for iter in range(self.args.local_ep):
                    batch_loss = []
                    for batch_idx, (images, labels) in enumerate(self.ldr_train):
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        net.zero_grad()
                        log_probs = net(images)
                        #print("label is :", labels, len(labels))
                        loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        optimizer.step()
                        if self.args.verbose and batch_idx % 10 == 0:
                            """
                            print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                    100. * batch_idx / len(self.ldr_train), loss.item()))
                            """
                        batch_loss.append(loss.item())
                    epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)