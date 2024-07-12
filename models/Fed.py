#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np

import math

from models.Function import CosSim, CosSim_with_key

'''
### #1. def FedAvg 
### #2. def FedAvg_w_Krum
### #3. def FedAvg_w_multiKrum
### #4. def FedAvg_w_median
### #5. def FedAvg_w_trimmedmean

### #6. def FedAvg_w_FoolsGold
### #7. def FedAvg_w_Auror
### #8. def FedAvg_w_AFA

### #9. def FedAvg_w_FLAME

### #n. class FedAvg_w_centralDB(object):
###
### But n is not used in original FL.
'''

def FedAvg(w, losses):
    w_avg = copy.deepcopy(w[0])
    print("len w is :", len(w))

    for k in w_avg.keys():
        for i in range(1, len(w)):
            if w_avg[k].dtype == w[i][k].dtype:
                w_avg[k] += w[i][k]
            else:
                #tmp = torch.tensor(w[i][k], dtype=w_avg[k].dtype)
                tmp = w[i][k].clone().detach()
                tmp = tmp.to(w_avg[k].dtype)
                w_avg[k] += tmp

        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg, losses


def FedAvg_ver2(w, w_glob, losses):
    w_avg = copy.deepcopy(w_glob)
    print("len w is :", len(w))

    for k in w_avg.keys():
        w_avg[k] -= w_glob[k]
        for i in range(0, len(w)):
            if w_avg[k].dtype == w[i][k].dtype:
                w_avg[k] += (w[i][k]- w_glob[k])
            else:
                #tmp = torch.tensor(w[i][k], dtype=w_avg[k].dtype)
                tmp = w[i][k].clone().detach()
                tmp = tmp.to(w_avg[k].dtype)
                w_avg[k] += (tmp- w_glob[k])

        w_avg[k] = torch.div(w_avg[k], len(w)/0.1)
        w_avg[k] += w_glob[k]
    
    return w_avg, losses

def FedAvg_w_multiKrum(w, losses, num_users, num_comps):
    #num_users = num_users; num_comps = num_comps ## n, f

    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])

    score = np.array([0.0 for i in range(len(w))])
    score_each = np.array([[0.0 for j in range(len(w))] for i in range(len(w))])

    tmp = copy.deepcopy(w)
    loss = copy.deepcopy(losses)

    ## l = n - f - 2
    l = num_users - num_comps - 2

    for k in w_avg.keys(): ## for each key
        for i in range(len(w)): ## for each local models
            w_tmp = 0
            A = tmp[i][k].cpu().numpy()
            for j in range(len(w)):
                if j == i:
                    continue
                else:
                    A = tmp[i][k].cpu().numpy()
                    B = tmp[j][k].cpu().numpy()
                    w_tmp = np.linalg.norm(A - B, ord=None)
                    #w_tmp = torch.cdist(w[i][k], w[j][k], p=2)
                    w_tmp = w_tmp ** 2
                score_each[i][j] += w_tmp;

    ## To extract l nearest vectors
    for i in range(len(w)):
        score_each[i] = np.sort(score_each[i])
        for j in range(l):
            score[i] += score_each[i][j+1]

    ## for global loss
    loss_tmp = [0.0 for i in range(l)]

    sorted_idx_score = np.argsort(score)

    print(sorted_idx_score)
    print()
    print(np.sort(score))

    #print("# of selected weights:", l)
    #print("original score is :", score)
    #print("sorted idx is :", sorted_idx_score)

    ### compute global weight
    for k in w_avg.keys():
        for i in range(l):
            #print("Correct :", sorted_idx_score[i])
            print(w_avg[k].dtype, w[sorted_idx_score[i]][k].dtype, k, np.shape(w[sorted_idx_score[i]][k])) 
            w_avg[k] += w[sorted_idx_score[i]][k]
        #print()
        w_avg[k] = torch.div(w_avg[k], l)

    ### compute global loss
    for i in range(l):
        loss_tmp[i] = loss[sorted_idx_score[i]]

    return w_avg, loss_tmp


def FedAvg_w_Krum(w, losses, num_users, m, num_comps):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])

    score = np.array([0.0 for i in range(len(w))])

    tmp = copy.deepcopy(w)
    loss = copy.deepcopy(losses)

    for k in w_avg.keys(): ## for each key
        for i in range(len(w)): ## for each local models
            w_tmp = 0
            A = tmp[i][k].cpu().numpy()
            for j in range(len(w)):
                if j == i:
                    continue
                else:
                    B = tmp[j][k].cpu().numpy()
                    w_tmp = np.linalg.norm(A - B, ord=None)
                    #w_tmp = torch.cdist(w[i][k], w[j][k], p=2)
                    w_tmp = w_tmp ** 2
                score[i] += w_tmp

    ## l = n - f - 2
    #l = num_users - num_comps - 2
    l = 1

    ## for global loss
    loss_tmp = [0.0 for i in range(l)]

    sorted_idx_score = np.argsort(score)

    #print(score)
    #print(sorted_idx_score)

    #print("# of selected weights:", l)
    #print("original score is :", score)
    #print("sorted idx is :", sorted_idx_score)

    ### compute global weight
    for k in w_avg.keys():
        for i in range(l):
            print("Correct :", sorted_idx_score[i])
            w_avg[k] += w[sorted_idx_score[i]][k]
        print()
        w_avg[k] = torch.div(w_avg[k], l)

    ### compute global loss
    for i in range(l):
        loss_tmp[i] = loss[sorted_idx_score[i]]

    return w_avg, loss_tmp, sorted_idx_score[0]


#### This is coordinate-wise median
def FedAvg_w_median(w, losses, num_users, m, num_comps):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])

    w_tmp = copy.deepcopy(w)

    for k in w_avg.keys(): ## for each key
        tmp = w_tmp[0][k]
        Dim = torch.Tensor.dim(tmp)
        for i in range(1, num_users):
            A = w_tmp[i][k]
            tmp = torch.cat([tmp, A], dim=0)

        if Dim == 3:
            tmp = torch.reshape(tmp, (num_users, w_avg[k].size(0), w_avg[k].size(1), w_avg[k].size(2)))
        elif Dim == 2:
            tmp = torch.reshape(tmp, (num_users, w_avg[k].size(0), w_avg[k].size(1)))
        else:
            tmp = torch.reshape(tmp, (num_users, w_avg[k].size(0)))

        #print("Dim is:", torch.Tensor.dim(tmp), "shape is :", tmp.shape)

        sorted, indices = torch.sort(tmp, dim=0)

        ### Select median value for each coordinate
        ### compute global weight
        if len(sorted) % 2 == 0:
            idx = len(sorted) // 2
            w_avg[k] = (sorted[idx - 1] + sorted[idx]) / 2
        else:
            idx = len(sorted) // 2
            w_avg[k] = sorted[idx]

    ### compute global loss
    ## for global loss
    loss_tmp = [0.0 for i in range(len(tmp))]
    losses = copy.deepcopy(torch.Tensor(losses))
    sorted, indices = torch.sort(losses, dim=0)

    ## compute global loss
    if len(sorted) % 2 == 0:
        idx = len(sorted) // 2
        loss_tmp = [(sorted[idx - 1] + sorted[idx]) / 2]
    else:
        idx = len(sorted) // 2
        loss_tmp = [sorted[idx]]

    return w_avg, loss_tmp

#### This is coordinate-wise trimmed mean
def FedAvg_w_trimmedmean(w, losses, num_users, m, num_comps):
    b = 0.4 ## hyperparameter, beta fraction
    T = int(b * num_users) ## as default, num_users = 10 so T is 2
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])

    w_tmp = copy.deepcopy(w)

    for k in w_avg.keys(): ## for each key
        #tmp = w_avg[k]
        tmp = w_tmp[0][k]
        Dim = torch.Tensor.dim(tmp)
        for i in range(1, num_users):
            A = w_tmp[i][k]
            tmp = torch.cat([tmp, A], dim=0)

        if Dim == 3:
            tmp = torch.reshape(tmp, (num_users, w_avg[k].size(0), w_avg[k].size(1), w_avg[k].size(2)))
        elif Dim == 2:
            tmp = torch.reshape(tmp, (num_users, w_avg[k].size(0), w_avg[k].size(1)))
        else: ## Dim is 1
            tmp = torch.reshape(tmp, (num_users, w_avg[k].size(0)))

        sorted, indices = torch.sort(tmp, dim=0)

        ### This should be zero for beta fraction both tail and head
        for i in range(T):
            sorted[i] = torch.zeros_like(sorted[i]) ## for tail
            sorted[(num_users - 1) - i] = torch.zeros_like(sorted[i]) ## for head

        ### Cutting torch with beta fraction
        ## for T
        tmp = copy.deepcopy(sorted[T])
        ## for from T + 1 to (N - T - 1)
        for i in range(T + 1, num_users - T):
            tmp = torch.cat([tmp, sorted[i]], dim=0)

        if Dim == 3:
            tmp = torch.reshape(tmp, (num_users - 2*T, w_avg[k].size(0), w_avg[k].size(1), w_avg[k].size(2)))
        elif Dim == 2:
            tmp = torch.reshape(tmp, (num_users - 2*T, w_avg[k].size(0), w_avg[k].size(1)))
        else: ## Dim is 1
            tmp = torch.reshape(tmp, (num_users - 2*T, w_avg[k].size(0)))

        ## l = m * (1 - 2b) = m - 2T
        ## in this example, l is 6
        l = len(tmp)

        ### compute global weight
        for i in range(l):
            w_avg[k] += tmp[i]
        w_avg[k] = torch.div(w_avg[k], l)

    ### compute global loss
    ## for global loss
    loss_tmp = [0.0 for i in range(l)]
    losses = copy.deepcopy(torch.Tensor(losses))
    sorted, indices = torch.sort(losses, dim=0)
    for i in range(T):
        sorted[i] = torch.zeros_like(sorted[i]) ## for tail
        sorted[(num_users - 1) - i] = torch.zeros_like(sorted[i]) ## for head

    sorted = sorted.tolist()

    ## Cutting torch with beta fraction
    ## compute global loss
    for i in range(l):
        loss_tmp[i] = sorted[i + T]

    return w_avg, loss_tmp

''' Insert the code here

### #6. def FedAvg_FoolsGold
### #7. def FedAvg_Auror
### #8. def FedAvg_AFA
### #9. def FedAvg_FLAME
'''

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

## For plotting for HDBSCAN
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
plt.rcParams["figure.figsize"] = [9,7]
"""!pip install hdbscan"""

def FedAvg_FLAME(w, w_glob, losses, num_users, m, num_comps):

   
    #Cosine distance for local weights
    score = CosSim(w)
    score = 1. - score
    weight = copy.deepcopy(w)
    weight_glob = copy.deepcopy(w_glob)
    loss = copy.deepcopy(losses)

    # To 2-norm 
    pca = PCA(2)
    transform = pca.fit_transform(score)
 
    # HDBSCAN model initailizing and clustering and Filfering? (-> cluster_selection_epsilon= 0.5)
    clustering = hdbscan.HDBSCAN(min_cluster_size=int((num_users/2) + 1),allow_single_cluster=True,cluster_selection_epsilon= 0.5,  min_samples=1, gen_min_span_tree=True )
    clustering.fit(transform)
    labels = clustering.labels_

    # get cluster label and Outlier label // filtering by HDBSCAN
    cluster_labels = clustering.labels_
    outlier_scores_ = clustering.outlier_scores_


    print("Results are :")
    print(cluster_labels)

    # Get the indices of outliers and inliers based on cluster_labels
    outlier_indices = np.where(cluster_labels == -1)
    inlier_indices = np.where(cluster_labels != -1)[0]
    w_inline = [weight[idx] for idx in inlier_indices]
    loss_inline = [loss[idx] for idx in inlier_indices]

    #weight euclidian distance and S_t // clipping
    tmp = copy.deepcopy(weight[0])
    total_norm = [0. for i in range(len(weight))]

    for k in tmp.keys(): ## for each key  

        for i in range(len(weight)):
          w_tmp = weight[i][k].cpu().numpy()
          w_glob_tmp = weight_glob[k].cpu().numpy()
          norm_w = np.linalg.norm(w_tmp - w_glob_tmp)
          total_norm[i] += norm_w
          
    norm_tmp = [total_norm[idx] for idx in inlier_indices]
    S_t = np.median(total_norm)
    gamma = [0. for i in range(len(w_inline))]
    W_C = [weight_glob for i in range(len(w_inline))]
    
    for idx in range(len(w_inline)):
        #print(norm_tmp[idx])
        gamma[idx] = S_t/norm_tmp[idx]
    
    for k in tmp.keys():
        for idx in range(len(w_inline)):
            W_C[idx][k] = weight_glob[k] + (w_inline[idx][k]-weight_glob[k])*min(1,gamma[idx])        
   
    
    #BA
    #predict_backdoors = 0 
    #BA_count =  sum(1 for element in cluster_labels if element in comps_lables)
    #for i in len(comps_list):
     # if cluster_labels[comps_list[i]  
    
      #  predict_backdoors += 1  
    
   # print("Backdoor Accuracy (BA): ",  predict_backdoors/len(comps_list) )
   
   
    #noising // Adaptive Noising
    noise_eps = 0.1 # privacy tolerance
    noise_delta = 0.05 # tnoise distribution control paramete
    noise_lambda = (1/noise_eps)* math.sqrt(2 * math.log(1.25/noise_delta) )
    noise_level = S_t*noise_lambda


    # Avg
    w_avg = copy.deepcopy(w[0])
    l = len(W_C)
    for k in w_avg.keys():
        for i in range(l):
            if i==0:
                w_avg = copy.deepcopy(W_C[i])
            #print("Correct :", sorted_idx_score[i])
            else:
                w_avg[k] += W_C[i][k]
        #print()
        w_avg[k] = torch.div(w_avg[k], l) + noise_level

 
    return w_avg, losses

def FedAvg_FoolsGold(w, w_glob, losses, m, lr):
    device = torch.device('cuda:{}'.format(3) if torch.cuda.is_available() else 'cpu')
    #initialize variables
    w_tmp = copy.deepcopy(w)
    w_avg = copy.deepcopy(w_glob)
    w_init = copy.deepcopy(w[0])
    n_clients = m
    maxcs = {} 
    wv = {}
    
    #caculate Cosign similarity
    cs = CosSim_with_key(w_tmp)
    #print(cs)
    for k in w_init.keys():
        #cs[k] = cs[k] - np.eye(n_clients) 
        maxcs[k] = np.max(cs[k], axis = 1)

    
    
    for k in w_init.keys():
        for i in range(n_clients):
            for j in range(n_clients):
                if i==j:
                    continue
                if maxcs[k][i] < maxcs[k][j]:
                        
                    cs[k][i][j] = cs[k][i][j]*maxcs[k][i] / maxcs[k][j]
    
    
    for k in w_init.keys():
        wv[k] = 1. - (np.max(cs[k], axis = 1))
        
    for k in w_init.keys():
        wv[k][wv[k]>1] = 1.
        wv[k][wv[k]<0] = 0.
        alpha = np.max(cs[k], axis =1)
        tmp = np.max(wv[k])
        if tmp == 0.:
            tmp = .01
        else: 
            continue
        wv[k] = wv[k]/ tmp  
        wv[k][(wv[k]==1.)] = .99
        wv[k] = (np.log(wv[k]/(1.-wv[k])) + 0.5)
        wv[k][((np.isinf(wv[k]) + wv[k]) > 1)] = 1.
        wv[k][(wv[k]<0)] = 0.
    #print(w_avg)
        
    for k in w_init.keys():
        WV = torch.from_numpy(wv[k]).to(device)
        for i in range(m):    
            w_avg[k] = w_avg[k].double() + ((w_tmp[i][k]-w_glob[k]).double()) * WV[i].double()
    #print(w_avg)    
    return w_avg, losses

'''
def FedAvg_FoolsGold(w, losses, num_users, m, num_comps):
    # score is the result for this function
    score = np.array([[0. for i in range(len(w))]
                     for j in range(len(w))])  # n by n matrix
    w_avg = copy.deepcopy(w[0])  # To use keys in dict in w
    norm_total = [0. for i in range(len(w))]  # Denominator for CosSim

    tmp = copy.deepcopy(w)  # To compute CosSim
    for k in w_avg.keys():  # for each key
        for i in range(len(w)):  # for each local models
            ## Compute Denominator ##
            A = tmp[i][k].cpu().numpy()
            tmp_A = np.linalg.norm(A, ord=None)  # l2-norm
            tmp_A = tmp_A ** 2
            norm_total[i] += tmp_A
            flat_A = A.flatten()

            ## Compute numerator ##
            for j in range(len(w)):
                if j < i or j == i:  # because score is symmetric matrix
                    continue
                else:
                    B = tmp[j][k].cpu().numpy()
                    flat_B = B.flatten()
                    # Insert each numerator value into score matrix first
                    score[i][j] += np.dot(flat_A, flat_B)

    # Make symmetric matrix
    # Because we cannot compute for j < i for better computational complexity(De-duplicating for computing)
    score += score.T - np.diag(score.diagonal())
    norm_total = np.sqrt(norm_total)

    for i in range(len(w)):
        for j in range(len(w)):
            if i == j:
                score[i][j] = 0.
            else:
                score[i][j] = score[i][j] / (norm_total[i] * norm_total[j])

    print(score)

    epsilon = 1e-5
    n = len(w)
    d= 784

    #cs = smp.cosine_similarity(w) - np.eye(10)
    cs = score
    # Pardoning: reweight by the max value seen
    maxcs = np.max(cs, axis=1) + epsilon
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)

    wv[(wv == 1)] = 0.99

    # Logit function
    wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    print(w)

    print(w_avg)
    w_to_update= w_avg['layer_input.weight'].detach().cpu().numpy()

    # Apply the weight vector on this delta
    delta = np.reshape(w_to_update, (n, d))

    return np.dot(delta.T, wv), loss
'''

class FedAvg_w_centralDB(object):
    def __init__(self, args, dataset=None, idxs=None, w_dict=None, w=None, losses=None, thres=None, \
                 num_users=None, m=None, num_comps=None, net_glob_local=None, dict_users=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train_central = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.central_bs, shuffle=True)

        #self.idx = idx
        self.w_dict = w_dict
        self.w = w
        self.losses = losses
        self.thres = thres
        self.num_of_comp = args.num_comps # the number of compromised workers
        self.num_comps = num_comps
        self.img_size = dataset[0][0].shape
        self.num_users = num_users

        self.net_glob_local = net_glob_local
        self.dict_users = dict_users ## This is just #datarow for each user for compute weight

    def train(self, net):
        epoch_loss = []
        #print(self.idx)

        w_dict_tmp = copy.deepcopy(self.w_dict)
        w_tmp = copy.deepcopy(self.w)
        net_original = copy.deepcopy(net.state_dict())
        net_original_local = copy.deepcopy(self.net_glob_local.state_dict())

        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # compute the median of clipping threshold
        clip_thres = np.median(self.thres)

        print("median is :", clip_thres)

        ##################################################################################################
        ##################################################################################################
        #####                                                                                        #####
        ##### How to implement adding noise : https://medium.com/pytorch/differential-privacy-series #####
        #####                                 -part-1-dp-sgd-algorithm-explained-12512c3959a3        #####
        #####                                                                                        #####
        ##### for inplace changes : https://discuss.pytorch.org/t/                                   #####
        #####                       nonetype-object-has-no-attribute-zero/61013                      #####
        #####                                                                                        #####
        ##################################################################################################
        ##################################################################################################
        sigma = self.args.sigma * self.args.clipping
        for iter in range(self.args.local_ep):
            batch_loss = []

            for p in net.parameters():
                p.accumulated_grads = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train_central):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                ##optimizer.step()

                ## Clipping
                clip_grad_norm_(net.parameters(), max_norm=self.args.clipping)

                ##optimizer.step()

                ## for inplace changes - https://discuss.pytorch.org/t/nonetype-object-has-no-attribute-zero/61013
                ## it's both sigma and clipping th.
                with torch.no_grad():
                    ## Do DP
                    for p in net.parameters():
                        noise = torch.normal(mean=0., std=sigma, size=p.shape).cuda()
                        noise = (1 / self.args.local_bs) * noise
                        p.grad += noise
                        #p.grad += torch.normal(mean=0., std=sigma, size=p.shape).cuda()

                        # This is what optimizer.step() does
                        #p = p - self.args.lr * p.grad
                        p.sub_(p.grad * self.args.lr)
                        p.grad.zero_()
                ''''''

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train_central.dataset),
                               100. * batch_idx / len(self.ldr_train_central), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        ## current net weights
        net_dict = copy.deepcopy(net.state_dict())

        ## for Server
        net_delta = copy.deepcopy(net_dict)
        for k in net_delta.keys():
            net_delta[k] = net_dict[k] - net_original[k]

        #### Error-based local selection ####
        with torch.no_grad():

            each_loss = []
            each_err = []
            net.eval()
            for iter in range(len(w_dict_tmp)):
                net.load_state_dict(w_dict_tmp[iter])

                # train and update
                #optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
                total = 0
                correct = 0
                accu = 0; loss = 0; err = 0
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss = Variable(loss, requires_grad=True)
                    loss.backward()
                    optimizer.step()

                    # https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
                    _, predicted = torch.max(log_probs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                #err = 100 - (100 * correct / total)
                err = 1. - (1. * correct / total)

                #err = err / self.num_users
                #loss = loss / self.num_users

                each_loss.append(loss.item())
                each_err.append(err)

            each_loss = np.array(each_loss)
            each_err = np.array(each_err)

        #print("central DB loss value :", each_loss)
        #print("central DB accu value :", each_err)

        #print(each_err.shape)

        #w_avg = copy.deepcopy(self.w[0])
        #loss = copy.deepcopy(self.losses)

        #w_avg = torch.zeros_like(self.w[0])
        #loss = torch.zeros_like(self.losses)

        #for k in w_avg.keys():
        #    w_avg[k] = torch.zeros_like(w_avg[k])

        #l = self.num_users - self.num_comps - 2
        l = self.num_users - 48 - 2 ### 48 is the number of max f
        loss_tmp = [0.0 for i in range(l)]

        ## using err
        sorted_idx_score = np.argsort(each_err); sorted_idx_test = np.sort(each_err)
        print(sorted_idx_score[:l])
        ## using loss
        sorted_idx_score_loss = np.argsort(each_loss); sorted_idx_test_loss = np.sort(each_loss)

        ## sorting test
        sorteds = np.sort(each_err); sorteds_loss = np.sort(each_loss)
        #print(sorteds)
        #####################################

        ### This is beta fraction for hybrid aggregation
        ### beta : central DB / 1 - beta : the others
        beta = 0.1
        sigma_local = self.args.sigma * clip_thres
        print("sigma local is :", sigma_local)
        print("dict_users : ", self.dict_users, type(self.dict_users))

        ### gradient clip and adding noise
        ### each compute delta_w / clip C
        for i in range(len(self.thres)):
            ## compute max
            C = max(1, (self.thres[i]/clip_thres))
            #print("C is :", C)
            for k in net_original.keys():
                w_tmp[i][k] = w_tmp[i][k] / C
                #### ... adding noise
                noise = torch.normal(mean=0., std=sigma_local, size=w_tmp[i][k].shape).cuda()
                noise = (1 / self.dict_users) * noise
                w_tmp[i][k] += noise
                #w_tmp[i][k] = (1 / self.dict_users) * w_tmp[i][k]
                ## net_original + self.w(is equal to net_delta)
                #w_tmp[i][k] = net_original[k] + w_tmp[i][k]

        #print("sorted idx is :", sorted_idx_score[0:l])

        w_avg = copy.deepcopy(w_tmp[sorted_idx_score[0]])
        w_avg_local = copy.deepcopy(w_avg)
        ### compute global weight
        for k in w_avg.keys():
            for i in range(1, l):
                w_avg[k] += w_tmp[sorted_idx_score[i]][k]
            w_avg[k] = torch.div(w_avg[k], l)
            w_avg_local[k] = ((1. - beta) * (net_original_local[k] + w_avg[k]))
            w_avg[k] = (beta * net_dict[k]) + w_avg_local[k]

        ### compute global loss
        for i in range(l):
            loss_tmp[i] = each_loss[sorted_idx_score[i]] ## for error
            #loss_tmp[i] = each_loss[sorted_idx_score_loss[i]] ## for loss

        return w_avg, w_avg_local, loss_tmp



