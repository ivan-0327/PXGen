import os
import sys


current_dir  = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from PXGen_utility.criterion_calculater import Criterion_Calculater
from Utility.file_utility import getfolderDataLoader_labelDic
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
from Model.VAE.model import loss_functionKLD,return_model ,loss_function

import torch.nn.functional as F
from torchvision.utils import save_image
from torch.optim import Adam

mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels =1 ),
])

'''For Mnist VAE Model'''
'''for stability of maximun boundary'''
'''iterate n times'''
def get_mean_maximun_boundary(model , generate_amount , iterating_times):
    g_max_KLD = 0
    g_max_Mse = 0
    for i in range(iterating_times):
        ''' generating samples with label 0'''
        #print(f' i:{i}')
        g_x =model.generate_batch_images(generate_amount)
        g_x_hat, g_mean, g_log_var = model(g_x)
        g_df_empty = pd.DataFrame({'Mse' : [] , 'KLD':[]})
        '''calculating KLD '''
        _ ,g_KLD   =loss_functionKLD( g_mean.cpu(), g_log_var.cpu())
        '''calculating Mse'''
        Mse_loss = F.mse_loss(g_x_hat.view(-1,784).cpu(), g_x.view(-1,784).cpu(), reduction='none')
        Mse_loss = torch.sum(Mse_loss,dim =1)  
        g_df_empty['KLD']              = g_KLD.detach().cpu().numpy()
        g_df_empty['Mse']              = Mse_loss.detach().cpu().numpy()

        KLD_countperset = int(g_df_empty['KLD'].count() *1)
        Mse_countperset = int(g_df_empty['Mse'].count() *1)

        sorted_dfKLD = g_df_empty.sort_values(by='KLD')
        sorted_dfKLD = sorted_dfKLD.reset_index(drop=True)
        kld_perset= sorted_dfKLD.loc[KLD_countperset-1, 'KLD']
        
        sorted_dfMse = g_df_empty.sort_values(by='Mse')
        sorted_dfMse = sorted_dfMse.reset_index(drop=True)
        Mseperset= sorted_dfMse.loc[Mse_countperset-1, 'Mse']

        g_max_KLD += kld_perset
        g_max_Mse += Mseperset

    g_max_KLD = g_max_KLD/iterating_times
    g_max_Mse =g_max_Mse/iterating_times

    return g_max_KLD , g_max_Mse


'''experiment 4 function '''
'''calculate the values relating Qurdrant Chart'''
def calculate_sum_Mse_KLD(row):
    return row['Mse'] + row['KLD']
def calculate_sum_Dssim_KLD(row):
    return row['Dssim'] + row['KLD']

def calculate_sum_Mse_KLD_quadrant2(row):
    return -row['Mse'] + row['KLD']
def calculate_sum_Mse_KLD_quadrant4(row):
    return row['Mse'] - row['KLD']

def calculate_sum_Dssim_KLD_quadrant2(row):
    return -row['Mse'] + row['KLD']
def calculate_sum_Dssim_KLD_quadrant4(row):
    return row['Mse'] - row['KLD']

'''
## process data to classify quadrant
To calssify the categories of quadrant
'''    
def classify_quadrant(row , kld_median , Dssim_median , Mse_median , type_name):
    KLD_flag = 0 if row['KLD'] < kld_median else 1
    
    if type_name == 'quadrant_Mse_KLD' :
        Mse_flag = 0 if row['Mse'] < Mse_median else 1
        if KLD_flag == 1 and Mse_flag == 1 :
            return 'LILE'
        elif KLD_flag == 1 and Mse_flag == 0 :
            return 'LIHE'
        elif KLD_flag == 0 and Mse_flag == 0 :
            return 'HIHE'
        elif KLD_flag == 0 and Mse_flag == 1 :
            return 'HILE'
    elif type_name == 'quadrant_Dssim_KLD' :
        Dssim_flag = 0 if row['Dssim'] < Dssim_median else 1
        if KLD_flag == 1 and Dssim_flag == 1 :
            return 1
        elif KLD_flag == 1 and Dssim_flag == 0 :
            return 2
        elif KLD_flag == 0 and Dssim_flag == 0 :
            return 3
        elif KLD_flag == 0 and Dssim_flag == 1 :
            return 4
        
        

def PXGen_classify_from_folder(data_path , given_model , DEVICE ,model_boundary_values_save_path):
    print(f'load data path  :{data_path}')
    data_loader , data_label_dic = getfolderDataLoader_labelDic(data_path , 50 ,mnist_transform)
    
    model    = given_model

    '''if there is a record , loading the values from record .'''
    if not os.path.exists(model_boundary_values_save_path+'/model_boundary.npy'):

        max_KLD ,max_Mse=get_mean_maximun_boundary(model , 60000 ,100)

        '''save g_max_KLD ,g_max_KLD'''
        if not os.path.exists(model_boundary_values_save_path):
            os.makedirs(model_boundary_values_save_path)
        dict ={'max_KLD':max_KLD , 'max_Mse':max_Mse}
        np.save(model_boundary_values_save_path+'/model_boundary.npy' ,dict )

    else:

        load_dict = np.load(model_boundary_values_save_path+'/model_boundary.npy',allow_pickle="TRUE").item()
        max_KLD ,max_Mse = load_dict['max_KLD'] , load_dict['max_Mse']

    print(f'max_KLD ,max_Mse :{max_KLD ,max_Mse}')
    '''參與訓練的資料'''
    Calculater = Criterion_Calculater(model ,data_loader , data_label_dic ,loss_functionKLD  ,DEVICE  )
    df ,x_df , x_hat_df= Calculater.Get_KLD_And_MSE_DSSIM()
    df['TrainOrExperiment'] ='Anchors'


    kld_boundary = max_KLD
    Mse_boundary = max_Mse

    df['quadrant_Mse_KLD'] = df.apply(classify_quadrant ,axis = 1 ,args=(kld_boundary , 0 , Mse_boundary,'quadrant_Mse_KLD')  )
    df['sum_Mse_KLD']      = df.apply(calculate_sum_Mse_KLD, axis=1)
    df['sum_Mse_KLD_quadrant2']      = df.apply(calculate_sum_Mse_KLD_quadrant2, axis=1)
    df['sum_Mse_KLD_quadrant4']      = df.apply(calculate_sum_Mse_KLD_quadrant4, axis=1)

    
    # load
    #load_dict = np.load(g_max_values_save_path+'/g_max.npy',allow_pickle="TRUE").item()
    #print(load_dict)
    return df ,x_df ,x_hat_df#.astype(float)