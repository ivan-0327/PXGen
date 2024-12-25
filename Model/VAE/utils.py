"""utils.py"""

import argparse
import os
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

kwargs = {'num_workers': 1, 'pin_memory': True}

def check_path_and_crete(path) :
    if not os.path.isdir(path) :
        os.makedirs(path)
        
def classify_save_image(path , images_all ,labels, number_array,train_data =True):
    if train_data :
        path =path+'/train'
    else:
        path =path+'/test'
    '''按照類型分類'''
    for number in number_array :
        '''更改路徑'''
        path_class = path+'/'+str(number)
        check_path_and_crete(path_class)
        #print(path_class)
        '''使用number建立filer'''
        image_filter = labels[labels[0] ==number]
                
        '''使用index ,一張一張儲存'''
        for index in image_filter.index :
            images = images_all.iloc[index].to_numpy()   
            images = torch.Tensor(images).view(1,28,28)
            '''檢查路徑是否存在'''
            
            '''存圖片'''
            save_path=path_class+'/'+str(index)+'.png'
            save_image(images,save_path)
    print('Finish!')
def cuda(tensor, uses_cuda, gpu):
    if not uses_cuda:
        return tensor
    device = torch.device("cuda:{}".format(gpu))
    return tensor.to(device)


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)



def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)
        
