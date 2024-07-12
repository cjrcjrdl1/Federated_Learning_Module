import copy
import math
import torch
from torch import nn
import numpy as np

#####################
##### Distances #####
#####################

## Cosine Similarity
def CosSim(w):
    # score is the result for this function
    score = np.array([[0. for i in range(len(w))] for j in range(len(w))]) ## n by n matrix
    w_avg = copy.deepcopy(w[0]) # To use keys in dict in w
    norm_total = [0. for i in range(len(w))] # Denominator for CosSim

    tmp = copy.deepcopy(w) # To compute CosSim
    for k in w_avg.keys(): ## for each key  
        for i in range(len(w)): ## for each local models
            ## Compute Denominator ##
            A = tmp[i][k].cpu().numpy()
            tmp_A = np.linalg.norm(A, ord=None) ## l2-norm
            #tmp_A = tmp_A ** 2
            norm_total[i] += tmp_A
            flat_A = A.flatten()

            ## Compute numerator ##
            for j in range(len(w)):
                if j < i or j == i: ## because score is symmetric matrix
                    continue
                else:
                    B = tmp[j][k].cpu().numpy()
                    flat_B = B.flatten()
                    score[i][j] += np.dot(flat_A, flat_B) # Insert each numerator value into score matrix first


    # Make symmetric matrix 
    # Because we cannot compute for j < i for better computational complexity(De-duplicating for computing)
    score += score.T - np.diag(score.diagonal())
    #norm_total = np.sqrt(norm_total)

    for i in range(len(w)):
        for j in range(len(w)):
            if i == j:
                score[i][j] = 1.
            else:
                score[i][j] = score[i][j] / (norm_total[i] * norm_total[j])

    #print(score)
    
    return score

def CosSim_with_key(w):
    # score is the result for this function
    score = {}
    norm_total = {}
    w_avg = copy.deepcopy(w[0]) # To use keys in dict in w
    for k in w_avg.keys():
        score[k] = np.array([[0. for i in range(len(w))] for j in range(len(w))]) ## n by n matrix
        norm_total[k] = [0. for i in range(len(w))] # Denominator for CosSim
    tmp_A = {}
    tmp = copy.deepcopy(w) # To compute CosSim
    for k in w_avg.keys(): ## for each key
        for i in range(len(w)): ## for each local models
            ## Compute Denominator ##
            A = tmp[i][k].cpu().numpy()
            tmp_A[k] = np.linalg.norm(A, ord=None) ## l2-norm
            #tmp_A = tmp_A ** 2
            norm_total[k][i] += tmp_A[k]
            flat_A = A.flatten()

            ## Compute numerator ##
            for j in range(len(w)):
                if j < i or j == i: ## because score is symmetric matrix
                    continue
                else:
                    B = tmp[j][k].cpu().numpy()
                    flat_B = B.flatten()
                    score[k][i][j] += np.dot(flat_A, flat_B) # Insert each numerator value into score matrix first


    # Make symmetric matrix 
    # Because we cannot compute for j < i for better computational complexity(De-duplicating for computing)
    for k in w_avg.keys():
        score[k] += score[k].T - np.diag(score[k].diagonal())
    #norm_total = np.sqrt(norm_total)
    for k in w_avg.keys():
        for i in range(len(w)):
            for j in range(len(w)):
                if i == j:
                    score[k][i][j] = 1.
                else:
                    score[k][i][j] = score[k][i][j] / (norm_total[k][i] * norm_total[k][j])

    #print(score)

    return score

def dist_norm(w_avg, model):
  dist = 0.
  w_glob = copy.deepcopy(w_avg)
  w_local = copy.deepcopy(model)
  norm_total_glob = 0.
  norm_total_local = 0.
  for k in w_avg.keys():
    w_g = w_glob[k].cpu().numpy()
    norm_glob = np.linalg.norm(w_g, ord=None)
    norm_glob = norm_glob**2
    w_l = w_glob[k].cpu().numpy()
    norm_local = np.linalg.norm(w_l, ord=None)
    norm_local = norm_local**2
    norm_total_glob +=norm_glob
    norm_total_local +=norm_local
  dist = norm_total_glob-norm_total_local
  dist = np.sqrt(dist)
  
  return dist
  
######################
##### Clustering #####
######################









