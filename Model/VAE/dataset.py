import os
import sys
# Get the parent directory
current_dir  = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
import torch
import gc
import random
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
#from utils import classify_save_image ,kwargs
from Model.VAE.utils import classify_save_image ,kwargs
dataset_path = os.path.join(current_dir ,'content/')
dataset_path_idx1 = os.path.join(current_dir ,'content/','MNIST','raw','train-labels-idx1-ubyte') 
dataset_path_idx3 = os.path.join(current_dir ,'content/','MNIST','raw','train-images-idx3-ubyte')  

'''手動處理資料集 '''
from typing import Tuple
import struct
from array import *

def read_imagelabel_in_ubyte(path:str) ->Tuple[int,int,pd.DataFrame]:
  file = open(path, mode ="rb")
  magic_number , count  = struct.unpack(">ii",file.read(8))
  labels =np.fromfile(file=file ,dtype =np.uint8)
  labels = pd.DataFrame(labels)
  return magic_number ,count ,labels

def read_image_in_ubyte(path:str) ->Tuple[int,int,int,int,pd.DataFrame]:
  file = open(path, mode ="rb")
  magic_number , count ,rows ,columns = struct.unpack(">iiii",file.read(16))
  images:np.array =np.fromfile(file=file , dtype =np.uint8)
  images =images.reshape(count , (rows*columns))
  images =pd.DataFrame(images)
  return magic_number , count ,rows ,columns ,images
def get_number_data(number_array:array ,label_path , images_path):
    labels = read_imagelabel_in_ubyte(label_path)[-1]
    images = read_image_in_ubyte(images_path)[-1]

    #make a filter
    #print(f'number_array :{number_array}')

    labels_filter = labels[labels[0].isin(number_array)]
    #print(f'labels_filter:{labels_filter}')
    labels = labels.iloc[labels_filter.index]
    images = images.iloc[labels_filter.index]

    labels =labels.reset_index(drop =True)
    images =images.reset_index(drop =True)
    return   images ,labels

def Produce_TrainingAndExperiment_Data_SingleType(SingleType_data ,SingleType_data_labels , used_Traing_number_perImages ):
    data_length = used_Traing_number_perImages#round(len(SingleType_data)* Traing_rate)

    '''Training data'''
    training_data = SingleType_data[:data_length]
    training_data_labels = SingleType_data_labels[:data_length]
    '''Experiment data'''
    experiment_data = SingleType_data[data_length:]
    experiment_data_labels = SingleType_data_labels[data_length:]
    
    return training_data , experiment_data , training_data_labels , experiment_data_labels

'''目的:將訓練以及實驗的資料分開，確定訓練資料的比例，剩下的多是實驗資料'''
'''訓練及實驗資料可設定圖片種類'''
def Produce_TrainingAndExperiment_Data(number_train_array:array ,number_experiement_array:array,label_path , images_path, used_Traing_number_perImages):
    '''每一種圖片遍歷'''
    for i in range(10):
        '''取得單一種圖片'''
        
        single_images, single_images_labels= get_number_data( [i],label_path,images_path)

        '''如果該種圖片為訓練所需則按比例分為訓練組跟實驗組'''
        '''如果不是訓練所需都分到實驗組'''
        if i in number_train_array:
            train_data , experiment_data , train_data_labels , experiment_data_labels = Produce_TrainingAndExperiment_Data_SingleType(single_images ,single_images_labels, used_Traing_number_perImages)
        else:
            train_data , experiment_data , train_data_labels , experiment_data_labels= Produce_TrainingAndExperiment_Data_SingleType(single_images ,single_images_labels , 0 )
        
        '''將訓練用資料整合'''

        if i == 0:
            training_images  =train_data
            training_images_labels  =train_data_labels
        else:
            training_images    = pd.concat([training_images,train_data],axis=0)
            training_images_labels = pd.concat([training_images_labels,train_data_labels],axis=0)

        '''將實驗用資料整合'''
        if i in number_experiement_array:
            if i == 0:
                experiement_images  =experiment_data
                experiment_images_labels  =experiment_data_labels
            else:
                experiement_images    = pd.concat([experiement_images,experiment_data],axis=0)
                experiment_images_labels =pd.concat([experiment_images_labels,experiment_data_labels],axis=0)

        training_images = training_images.reset_index(drop=True)
        experiement_images = experiement_images.reset_index(drop=True)
        training_images_labels = training_images_labels.reset_index(drop=True)
        experiment_images_labels = experiment_images_labels.reset_index(drop=True)

    return training_images , experiement_images , training_images_labels , experiment_images_labels



'''將處理好的資料做成DataSet與DataLoader'''

from torch.utils.data import Dataset , DataLoader
from sklearn.preprocessing import OneHotEncoder ,StandardScaler,LabelEncoder
class Custimom_DataSet(Dataset):
    def __init__(self,DataFrame_image , transform, mode ="train"):

        self.mode =mode
        self.data = DataFrame_image.to_numpy()#.astype(float)
        self.transform = transform


        if mode =="train":

            indices =[ i for i in range(len(self.data)) if i % 10 != 0]
            self.data = torch.from_numpy(self.data[indices])
            self.dim = self.data.shape[1]
            print(f'Finishing {mode} DataSet ,There are {len(self.data)} smaples (each dim  = {self.dim}) ')
        elif mode =="test":
            indices =[ i for i in range(len(self.data)) if i % 10 == 0]
            self.data = torch.from_numpy(self.data[indices])
            self.dim = self.data.shape[1]
            print(f'Finishing {mode} DataSet ,There are {len(self.data)} smaples (each dim  = {self.dim}) ')
        else:
             
            self.data = torch.from_numpy(self.data)
            self.dim = self.data.shape[1]
            print(f'Finishing {mode} DataSet ,There are {len(self.data)} smaples (each dim  = {self.dim}) ')

    def __len__(self ):

        return len(self.data)

    def __getitem__(self , index):

        return (self.data[index].float())/255
    


def return_data(args):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    train_number =args.train_number
    experiement_number =[0,1,2,3,4,5,6,7,8,9]
    training_images , experiement_images , training_images_labels ,experiment_images_labels =\
        Produce_TrainingAndExperiment_Data(train_number ,experiement_number, dataset_path_idx1  ,dataset_path_idx3 , args.train_amount )
    
    '''存訓練資料'''
    classify_save_image(args.save_traindata_dir+'data/label_'+str(train_number)+'Model', training_images ,training_images_labels ,train_number)
    '''存實驗資料 become the same with training data'''
    classify_save_image(args.save_traindata_dir+'data/label_'+str(train_number)+'Model', experiement_images ,experiment_images_labels ,experiement_number ,False)
    
    '''DataSet'''
    imageDataSet_Training = Custimom_DataSet(training_images, transform=mnist_transform , mode = "none")
    
    
    '''DataLoader'''
    train_loader = DataLoader(dataset=imageDataSet_Training, batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader

def Process_AnchorSet(args):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    train_number =args.train_number
    experiement_number =[0,1,2,3,4,5,6,7,8,9]
    training_images , experiement_images , training_images_labels ,experiment_images_labels =\
        Produce_TrainingAndExperiment_Data(train_number ,experiement_number, args.dataset_path_idx1  ,args.dataset_path_idx3 , args.train_amount )
    
    '''存訓練資料'''
    classify_save_image(args.load_from_AnchorSet_dir+'data', training_images ,training_images_labels ,train_number)
    '''存實驗資料 become the same with training data'''
    classify_save_image(args.load_from_AnchorSet_dir+'data', experiement_images ,experiment_images_labels ,experiement_number ,False)
    
    

     


def return_data_experiment(args):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    train_number =args.train_number
    experiement_number =[0,1,2,3,4,5,6,7,8,9]
    training_images , experiement_images , training_images_labels ,experiment_images_labels =\
        Produce_TrainingAndExperiment_Data(train_number ,experiement_number, dataset_path_idx1  ,dataset_path_idx3 , args.train_amount )
    
    '''存訓練資料'''
    classify_save_image(args.save_traindata_dir+'data/label_'+str(train_number)+'Model', training_images ,training_images_labels ,train_number)
    '''存實驗資料 become the same with training data'''
    classify_save_image(args.save_traindata_dir+'data/label_'+str(train_number)+'Model', experiement_images ,experiment_images_labels ,experiement_number ,False)
    
    '''DataSet'''
    imageDataSet_Training = Custimom_DataSet(training_images, transform=mnist_transform , mode = "none")
    
    
    '''DataLoader'''
    train_loader = DataLoader(dataset=imageDataSet_Training, batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader , training_images , experiement_images , training_images_labels ,experiment_images_labels


'''讀取  get exist data from folder
   資料'''
def getfolderDataLoader_labelDic(path , batch_size,mnist_transform):
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    '''讀取資料夾圖片'''
    dataset = ImageFolder(path,transform =mnist_transform )
    loader  = DataLoader(dataset=dataset,  batch_size=batch_size, shuffle=False, **kwargs)
    
    '''反轉dic'''
    label_dic = {v : k for k,v  in dataset.class_to_idx.items() }
    
    return loader , label_dic