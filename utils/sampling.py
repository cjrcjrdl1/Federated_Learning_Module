#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import copy
poison_images = [330, 568, 3934, 12336, 30560]
poison_images_test = [30696, 33105, 33615, 33907, 36848, 40713, 41706]

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
    
def DBA_testdata(dataset, poison_label):
    gpu = 3
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and \
                               gpu != -1 else 'cpu')
    testdata = []
    for idx , x in enumerate(dataset):
        _, label = x
        if label != poison_label:
            testdata.append(idx)
    testdataset = DataLoader(DatasetSplit(dataset, testdata), batch_size=len(testdata), shuffle=True, \
                                    num_workers=2, persistent_workers=True)

    global_patterns = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 10], [0, 11], [0, 12], [0, 13], [4, 1], [4, 2], [4, 3], [4, 4], [4, 10], [4, 11], [4, 12], [4, 13]]
    local1_patterns = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    local2_patterns = [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
    local3_patterns = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
    local4_patterns = [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]

    for batch_idx, (batch) in enumerate(testdataset): ####### global trigger
        images, targets = batch

        global_images=copy.deepcopy(images)
        global_targets=copy.deepcopy(targets)
        for index in range(0, len(images)):
            global_targets[index] = poison_label
            for i in range(0,len(global_patterns)):
                pos = global_patterns[i]
                global_images[index][0][pos[0]][pos[1]] = 1
                global_images[index][1][pos[0]][pos[1]] = 1
                global_images[index][2][pos[0]][pos[1]] = 1

            #global_images = global_images.to(device)
            #global_targets = global_targets.to(device)

        global_images.requires_grad_(False)
        global_targets.requires_grad_(False)
    global_set = list(zip(global_images, global_targets))
    #global_set = TensorDataset((global_images, global_targets))
##########################################################################################
    for batch_idx, (batch) in enumerate(testdataset): ####### local1 trigger
        images, targets = batch

        local1_images=copy.deepcopy(images)
        local1_targets=copy.deepcopy(targets)
        for index in range(0, len(images)):
            local1_targets[index] = poison_label
            for i in range(0,len(local1_patterns)):
                pos = local1_patterns[i]
                local1_images[index][0][pos[0]][pos[1]] = 1
                local1_images[index][1][pos[0]][pos[1]] = 1
                local1_images[index][2][pos[0]][pos[1]] = 1

            #local1_images = local1_images.to(device)
            #local1_targets = local1_targets.to(device)

        local1_images.requires_grad_(False)
        local1_targets.requires_grad_(False)
    local1_set = list(zip(local1_images, local1_targets))
    #local1_set = TensorDataset((local1_images, local1_targets))
##########################################################################################
    for batch_idx, (batch) in enumerate(testdataset): ####### local2 trigger
        images, targets = batch

        local2_images=copy.deepcopy(images)
        local2_targets=copy.deepcopy(targets)
        for index in range(0, len(images)):
            local2_targets[index] = poison_label
            for i in range(0,len(local2_patterns)):
                pos = local2_patterns[i]
                local2_images[index][0][pos[0]][pos[1]] = 1
                local2_images[index][1][pos[0]][pos[1]] = 1
                local2_images[index][2][pos[0]][pos[1]] = 1

            #local2_images = local2_images.to(device)
            #local2_targets = local2_targets.to(device)

        local2_images.requires_grad_(False)
        local2_targets.requires_grad_(False)
    local2_set = list(zip(local2_images, local2_targets))
    #local2_set = TensorDataset((local2_images, local2_targets))
##########################################################################################
    for batch_idx, (batch) in enumerate(testdataset): ####### local3 trigger
        images, targets = batch

        local3_images=copy.deepcopy(images)
        local3_targets=copy.deepcopy(targets)
        for index in range(0, len(images)):
            local3_targets[index] = poison_label
            for i in range(0,len(local3_patterns)):
                pos = local3_patterns[i]
                local3_images[index][0][pos[0]][pos[1]] = 1
                local3_images[index][1][pos[0]][pos[1]] = 1
                local3_images[index][2][pos[0]][pos[1]] = 1

            #local3_images = local3_images.to(device)
            #local3_targets = local3_targets.to(device)

        local3_images.requires_grad_(False)
        local3_targets.requires_grad_(False)
    local3_set = list(zip(local3_images, local3_targets))
    #local3_set = TensorDataset((local3_images, local3_targets))
##########################################################################################
    for batch_idx, (batch) in enumerate(testdataset): ####### local4 trigger
        images, targets = batch

        local4_images=copy.deepcopy(images)
        local4_targets=copy.deepcopy(targets)
        for index in range(0, len(images)):
            local4_targets[index] = poison_label
            for i in range(0,len(local4_patterns)):
                pos = local4_patterns[i]
                local4_images[index][0][pos[0]][pos[1]] = 1
                local4_images[index][1][pos[0]][pos[1]] = 1
                local4_images[index][2][pos[0]][pos[1]] = 1

            #local4_images = local4_images.to(device)
            #local4_targets = local4_targets.to(device)

        local4_images.requires_grad_(False)
        local4_targets.requires_grad_(False)
    local4_set = list(zip(local4_images, local4_targets))
    #local4_set = TensorDataset((local4_images, local4_targets))


    return global_set, local1_set, local2_set, local3_set, local4_set

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    #print(type(dict_users), dict_users[0], len(dict_users[0]))
    #print(type(dict_users[0]))
    return dict_users

from collections import defaultdict
import random

### dist using dirichlet prob (used in CaS paper)
def cifar_prop_dirichlet(dataset, num_users, prob):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
    """
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if ind in poison_images or ind in poison_images_test:
            continue
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    class_size = len(cifar_classes[0])
    per_participant_list = defaultdict(list)
    #per_participant_list = {i: np.array([], dtype='int64') for i in range(num_users)}
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
                np.array(num_users * [prob]))
        for user in range(num_users):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

    per_participant_list = dict(per_participant_list)
    for i in range(len(per_participant_list)):
        per_participant_list[i] = set(per_participant_list[i])

    return per_participant_list

### I think we need to create new one with prob. p
def prop_noniid(dataset, num_users, prob):
    """ Our code
    Sample non-IID client data from MNIST dataset
    with prob. p which as described in Usenix paper
    """
    print("non-IID with prob. p setting is selected")
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    #dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    idxs = np.arange(len(dataset))

    #labels = dataset.targets.numpy()
    #print(type(dataset.targets)) ## list
    labels = np.array(dataset.targets)

    # from 0 ~ 9999 is central DB, the others training datasets
    # about Central DB when only not 0
    # 0 means fully distributed FL (not Blended FL)
    num_items = 0
    labels = labels[num_items:]
    central_DB = idxs[:num_items]

    idxs = np.arange(len(labels))

    print("ffffff :", central_DB)
    print(labels)

    ###### Parameter settings ######

    ## the number of labels : we should modify later as automatically
    l = 10

    ## the number of users in each labels
    ## We assume the there is no float, i.e., num_users % l = 0
    n = int(num_users / l)
    print("n is num_users / l :", n)

    ## Datasize for each local users
    #d = int(len(dataset) / num_users)
    d = int(len(labels) / num_users)

    ## the number of fixed data with prob.
    fixed_d = int(d * prob)
    randomly_d = int(d * (1 - prob))
    print("randomly_d is :", fixed_d, randomly_d)

    ################################

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]     # idx for datarow
    idxs2 = idxs_labels[1,:]    # label value

    # divide num_users using by sorted labels
    idxs_labels_total = [[] for i in range(l)]
    idxs_tmp = [[] for i in range(l)]
    labels_tmp = [[] for i in range(l)]

    for i in range(len(idxs_labels_total)):
        for j in range(len(idxs)):
            if idxs2[j] == i:
                idxs_tmp[i].append(idxs[j])
                labels_tmp[i].append(idxs2[j])
        idxs_tmp[i] = np.array(idxs_tmp[i])
        labels_tmp[i] = np.array(labels_tmp[i])
        idxs_labels_total[i] = np.vstack((idxs_tmp[i], labels_tmp[i]))

        ''' # of datarows are equally divided for all labels for cifar dataset '''
        #print("###### : ", i)
        #print(idxs_labels_total[i].shape)
        #print(len(idxs_labels_total[i][0]))

    #print("total : ", np.shape(idxs_labels_total)) ## (10, 2, 5000)

    # divide datasets
    idx_users = [i for i in range(num_users)]
    rand_label = np.random.choice(l, l, replace=False)
    rand_user = np.random.choice(num_users, num_users, replace=False)
    #rand_user = np.split(rand_user, l)
    #print(len(idxs_labels_total[0][0]))

    #print("rand_label: ", rand_label)
    #print("rand_user: ", rand_user)
    
    remain_labels = [[] for i in range(l)]
    #### extract in fixed label with prob. p (fixed) ####
    for i in range(len(rand_label)):
        idx_label = rand_label[i]
        idxs = idxs_labels_total[idx_label][0, :]
        #print("idxs : ", idxs, len(idxs), i, rand_label[i])

        for j in range(n):
            idx_user = rand_user[i*n + j]
            if len(idxs) >= fixed_d: ## extract in fixed label with prob. p
                #print("len(idxs) : ", len(idxs))
                tmp = idxs[:fixed_d]
                dict_users[idx_user] = np.concatenate((list(dict_users[idx_user]), tmp), axis=0)
                dict_users[idx_user] = dict_users[idx_user].astype(int)
                dict_users[idx_user] = set(dict_users[idx_user])
                idxs = list(set(idxs) - set(tmp))
            else:
                tmp = idxs[:]
                dict_users[idx_user] = np.concatenate((list(dict_users[idx_user]), tmp), axis=0)
                dict_users[idx_user] = set(dict_users[idx_user])
                dict_users[idx_user] = set(dict_users[idx_user])

            #print(len(idxs))

        remain_labels[idx_label] = idxs

    print("after idxs_labels_total :", np.shape(idxs_labels_total), np.shape(remain_labels))

#    print("eeeeee :")
#    for i in range(len(dict_users)):
#        print(len(dict_users[i]), i)

    #print("sdfsdfsdf : ", np.shape(idxs_labels_total))
        
    #print("total : ", len(dict_users))
    #for i in range(len(dict_users)):
    #    print(len(dict_users[i]), i)
    ####
    
    idxs_remain = []
    for i in range(len(rand_label)):
        idx_label = rand_label[i]
        while (idx_label == rand_label[i]):
            idx_label = np.random.choice(rand_label, 1, replace=True)
            idx_label = idx_label[0]

        print(idx_label, rand_label[i])

        ## idxs_labels_total has the number of datarow for each label
        idxs = remain_labels[idx_label]
        if len(idxs) == 0:
            continue
        for j in range(n):
            idx_user = rand_user[i*n + j]
            while( len(list(dict_users[idx_user])) != d ):
                #tmp = idxs[:1]
                tmp = np.random.randint(len(idxs), size=1)
                dict_users[idx_user] = np.concatenate((list(dict_users[idx_user]), tmp), axis=0)
                dict_users[idx_user] = dict_users[idx_user].astype(int)
                dict_users[idx_user] = set(dict_users[idx_user])
            
                idxs = list(set(idxs) - set(tmp))

                #if len(dict_users[idx_user]) == d and len(idxs) != 0:
                if len(idxs) == 0:
                    break
                else:
                    for k in range(len(idxs)):
                        idxs_remain.append(idxs[k])

                idx_label = np.random.choice(rand_label, 1, replace=True)
                idx_label = idx_label[0]
                #idxs = idxs_labels_total[idx_label]
                #idxs = remain_labels[idx_label]
                remain_labels[idx_label] = idxs
                #print(np.shape(remain_labels))

        #idxs_labels_total[idx_label] = idxs
        #remain_labels[idx_label] = idxs
    ## for remaining idxs
    for idx_user in range(num_users):
        if len(dict_users[idx_user]) != d:
            sec = d - len(dict_users[idx_user])
            tmp = idxs_remain[:sec]
            dict_users[idx_user] = np.concatenate((list(dict_users[idx_user]), tmp), axis=0)
            dict_users[idx_user] = dict_users[idx_user].astype(int)
            dict_users[idx_user] = set(dict_users[idx_user])
            idxs_remain = list(set(idxs_remain) - set(tmp))

    ####
    for i in range(len(dict_users)):
        print("before :", len(dict_users[i]))
        dict_users[i] = set(dict_users[i])
        print("after :", len(dict_users[i]))

    #print("sdfsdf :", dict_users); print(len(dict_users[0]))
#    print(type(dict_users), dict_users[0], len(dict_users[0]))
    #print(type(dict_users[0]))

    return dict_users


def temp_prop_noniid(dataset, num_users, prob):
    """ Our code
    Sample non-IID client data from MNIST dataset
    with prob. p which as described in Usenix paper
    """
    print("non-IID with prob. p setting is selected")
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))

    #labels = dataset.targets.numpy()
    #print(type(dataset.targets)) ## list
    labels = np.array(dataset.targets)

    # from 0 ~ 9999 is central DB, the others training datasets
    # about Central DB when only not 0
    # 0 means fully distributed FL (not Blended FL)
    num_items = 0
    labels = labels[num_items:]
    central_DB = idxs[:num_items]

    idxs = np.arange(len(labels))

    print("ffffff :", central_DB)
    print(labels)

    ###### Parameter settings ######

    ## the number of labels : we should modify later as automatically
    l = 10

    ## the number of users in each labels
    ## We assume the there is no float, i.e., num_users % l = 0
    n = int(num_users / l)
    print("n is num_users / l :", n)

    ## Datasize for each local users
    #d = int(len(dataset) / num_users)
    d = int(len(labels) / num_users)

    ## the number of fixed data with prob.
    fixed_d = int(d * prob)
    randomly_d = int(d * (1 - prob))
    print("randomly_d is :", fixed_d, randomly_d)

    ################################

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]     # idx for datarow
    idxs2 = idxs_labels[1,:]    # label value

    # divide num_users using by sorted labels
    idxs_labels_total = [[] for i in range(l)]
    idxs_tmp = [[] for i in range(l)]
    labels_tmp = [[] for i in range(l)]

    for i in range(len(idxs_labels_total)):
        for j in range(len(idxs)):
            if idxs2[j] == i:
                idxs_tmp[i].append(idxs[j])
                labels_tmp[i].append(idxs2[j])
        idxs_tmp[i] = np.array(idxs_tmp[i])
        labels_tmp[i] = np.array(labels_tmp[i])
        idxs_labels_total[i] = np.vstack((idxs_tmp[i], labels_tmp[i]))

        ''' # of datarows are equally divided for all labels for cifar dataset '''
        #print("###### : ", i)
        #print(idxs_labels_total[i].shape)
        #print(len(idxs_labels_total[i][0]))

    #print("total : ", np.shape(idxs_labels_total)) ## (10, 2, 5000)

    # divide datasets
    idx_users = [i for i in range(num_users)]
    rand_label = np.random.choice(l, l, replace=False)
    rand_user = np.random.choice(num_users, num_users, replace=False)
    #rand_user = np.split(rand_user, l)
    #print(len(idxs_labels_total[0][0]))

    #print("rand_label: ", rand_label)
    #print("rand_user: ", rand_user)

    #### extract in fixed label with prob. p (fixed) ####
    for i in range(len(rand_label)):
        idx_label = rand_label[i]
        idxs = idxs_labels_total[idx_label][0, :]
        #print("idxs : ", idxs, len(idxs), i, rand_label[i])

        for j in range(n):
            idx_user = rand_user[i*n + j]
            if len(idxs) >= fixed_d: ## extract in fixed label with prob. p
                #print("len(idxs) : ", len(idxs))
                tmp = idxs[:fixed_d]
                dict_users[idx_user] = np.concatenate((dict_users[idx_user], tmp), axis=0)
                idxs = list(set(idxs) - set(tmp))
            else:
                tmp = idxs[:]
                dict_users[idx_user] = np.concatenate((dict_users[idx_user], tmp), axis=0)

            #print(len(idxs))

        idxs_labels_total[idx_label] = idxs

    #print("sdfsdfsdf : ", np.shape(idxs_labels_total))
        
    #print("total : ", len(dict_users))
    #for i in range(len(dict_users)):
    #    print(len(dict_users[i]), i)
    ####

    #### extract in the other label with prob. (1-p) (random) ####
    idxs_remain = []
    for i in range(len(rand_label)):
        idx_label = rand_label[i]
        while (idx_label == rand_label[i]):
            idx_label = np.random.choice(rand_label, 1, replace=True)
            idx_label = idx_label[0]

        ## idxs_labels_total has the number of datarow for each label
        idxs = idxs_labels_total[idx_label]
        if len(idxs) == 0:
            continue
        for j in range(n):
            idx_user = rand_user[i*n + j]
            while( len(dict_users[idx_user]) != d ):
                tmp = idxs[:1]
                dict_users[idx_user] = np.concatenate((dict_users[idx_user], tmp), axis=0)
                #dict_users[idx_user] = dict_users[idx_user].astype(int)
                idxs = list(set(idxs) - set(tmp))

                #if len(dict_users[idx_user]) == d and len(idxs) != 0:
                if len(idxs) == 0:
                    break
                else:
                    for k in range(len(idxs)):
                        idxs_remain.append(idxs[k])

                idx_label = np.random.choice(rand_label, 1, replace=True)
                idx_label = idx_label[0]
                idxs = idxs_labels_total[idx_label]

        idxs_labels_total[idx_label] = idxs

    ## for remaining idxs
    for idx_user in range(num_users):
        if len(dict_users[idx_user]) != d:
            sec = d - len(dict_users[idx_user])
            tmp = idxs_remain[:sec]
            dict_users[idx_user] = np.concatenate((dict_users[idx_user], tmp), axis=0)
            #dict_users[idx_user] = dict_users[idx_user].astype(int)
            idxs_remain = list(set(idxs_remain) - set(tmp))


    ####
    for i in range(len(dict_users)):
        print("before :", len(dict_users[i]))
        dict_users[i] = set(dict_users[i])
        print("after :", len(dict_users[i]))

    #print("sdfsdf :", dict_users); print(len(dict_users[0]))
    print(type(dict_users), dict_users[0], len(dict_users[0]))
    print(type(dict_users[0]))

    return dict_users



if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
