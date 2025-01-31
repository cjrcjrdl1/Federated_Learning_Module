import torch
from utils.options import args_parser
args = args_parser()
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and \
                               args.gpu != -1 else 'cpu')
#device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD='foolsgold'
MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
patience_iter=20

TYPE_LOAN='loan'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'