import argparse
import os
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
def get_ordered_checkpoint_list(path, suffix='', last_epoch=False):
    all_files = os.listdir(os.path.join(path))
    all_checkpoints = [f for f in all_files if f.endswith(suffix)]
    all_checkpoints.sort(key=int)
    print('{} checkpoints found'.format(len(all_checkpoints)))
    if not last_epoch:
        return all_checkpoints
    else:
        print('using the last epoch only')
        return [all_checkpoints[-1]]
    
'''load data from folder '''
def getfolderDataLoader_labelDic(path , batch_size,mnist_transform):
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    '''讀取資料夾圖片'''
    dataset = ImageFolder(path,transform =mnist_transform )
    loader  = DataLoader(dataset=dataset,  batch_size=batch_size, shuffle=False, **kwargs)
    
    '''反轉dic'''
    label_dic = {v : k for k,v  in dataset.class_to_idx.items() }
    
    return loader , label_dic

'''根據dataFrame save image'''
def save_image_for_quadrant_data(images_all ,labels ,path ):

    if not os.path.exists(path):
        os.makedirs(path)

    '''使用index ,一張一張儲存'''
    for index in labels.index :
        images = images_all.iloc[index].to_numpy()   
        images = torch.Tensor(images).view(1,28,28)
        '''檢查路徑是否存在'''
            
        '''存圖片'''
        save_path=path+'/'+str(index)+'.png'
        save_image(images,save_path)



def classify_save_image_for_quadrant_data(path , images_all ,labels, number_array, train_data =True):
    if train_data :
        path =path+'/train'
    else:
        path =path+'/test'
    '''按照類型分類'''
    for number in number_array :
        '''更改路徑'''
        path_class = path+'/'+str(number)
        
        
        #print(f'path_class:{path_class}')
        '''使用number建立filer'''
         
        image_filter = labels[labels['label'] ==str(number)]
        if len(image_filter) != 0:
            if not os.path.exists(path_class):
                os.makedirs(path_class)
            
        
        #print(f'number:{number}')     
        '''使用index ,一張一張儲存'''
        for index in image_filter.index :
            images = images_all.iloc[index].to_numpy()   
            images = torch.Tensor(images).view(1,28,28)
            '''檢查路徑是否存在'''
            
            '''存圖片'''
            save_path=path_class+'/'+str(index)+'.png'
            save_image(images,save_path)
    print(path+' : Finish!')

def gradually_reduce_data_And_save(images_all , labels , path = './data/only_quadrant_HIHE',include_flag =True , random_flag=False \
                                   ,define_amount_flag =False ,define_amount =1050):
    '''amount is the total number of reducing'''
    '''because the frequency of occurrences in quadrants LILE,LIHE,HILE is significantly lower than in quadrant HIHE,
       we set the reduction amount  to be the same as amount of quadrant LILE,LIHE,HILE . 
       
       1. reducing quadrant LILE,LIHE,HILE data from highest KLD+MSE model.
       2. reducing quadrant HIHE data from lowest KLD+MSE model .
       3. reducing random data
       
    '''
    total_amount= len(labels['quadrant_Mse_KLD'] )
    amount = len(labels[labels['quadrant_Mse_KLD'] !=str('HIHE')])
    
    print(f'amount:{amount}')

    if random_flag == False :
        for i in range(11):
        
            
            save_path= path +'/reduce_'+str(i*10)+'%'
            
            '''calculate the amount of reducing '''
            reduce_n = amount *(i/10)
            
            # 随机选择
            '''To archive to quadrant HIHE , select others'''
            if include_flag :
                image_filter = labels[labels['quadrant_Mse_KLD'] !=str('HIHE')]
                image_filter =image_filter.sort_values(by='sum_Mse_KLD', ascending=False)
            else :
                image_filter = labels[labels['quadrant_Mse_KLD'] ==str('HIHE')]
                image_filter =image_filter.sort_values(by='sum_Mse_KLD', ascending=True)
                
            rows = image_filter.head(int(reduce_n))
            
            residual_labels =labels.drop(rows.index)
            
            classify_save_image_for_quadrant_data( save_path, images_all ,residual_labels ,[0,1,2,3,4,5,6,7,8,9])
    else:
        if define_amount_flag  ==False :
            amount = len(labels[labels['quadrant_Mse_KLD'] !=str('HIHE')])
        else :
            amount = define_amount
        
        for i in range(11):
        
            
            save_path= path +'/reduce_'+str(i*10)+'%'
            
            '''calculate the amount of reducing '''
            reduce_n = amount *(i/10)

            
            # 随机选择
            '''To archive to quadrant 3 , select others'''
                   
            rows = labels.sample(n=int(reduce_n))          
            residual_labels =labels.drop(rows.index)
            
            classify_save_image_for_quadrant_data( save_path, images_all ,residual_labels ,[0,1,2,3,4,5,6,7,8,9])




