
import pandas as pd
import torch
from torch.nn import functional as F
def Calculate_Distance_Dssim(recurrent_index , images ,dispersion_images_x, calculate_distance_function):

    recurrent_image=images[recurrent_index]
    
    dispersion_images_x = dispersion_images_x.view( -1,1,28, 28)
    recurrent_image =recurrent_image.repeat(dispersion_images_x.shape[0],1,1,1).view( -1,1,28, 28)
    

    #print(f'recurrent_image :{recurrent_image.shape}')
    #print(f'images :{images.shape}')
    #計算距離 ssim 完全一樣值為1  ,1-ssim -> 0完全一樣 ,1完全不一樣
    #print(f'recurrent_image.shape:{recurrent_image.shape}')
    #print(f'dispersion_images_x.shape:{dispersion_images_x.shape}')
    min_distance_list =   (1-calculate_distance_function(recurrent_image , dispersion_images_x))/2 #   1-abs(calculate_distance_function(recurrent_image , images)) 
    
    #print(f'recurrent_index :{recurrent_index}')
    min_distance = min_distance_list.min()

    #print(f'i :{recurrent_index} min_distance :{min_distance}')
    return min_distance
def Calculate_Distance_Mse(recurrent_index , images ,dispersion_images_x):

    recurrent_image=images[recurrent_index]
    
    dispersion_images_x = dispersion_images_x.view( -1,1,28, 28)
    recurrent_image =recurrent_image.repeat(dispersion_images_x.shape[0],1,1,1).view( -1,1,28, 28)
    

    Mse_loss = F.mse_loss(recurrent_image.view(-1,784).cpu(), dispersion_images_x.view(-1,784).cpu(), reduction='none')
                          
    Mse_loss = torch.sum(Mse_loss,dim =1) 
    Mse_distance_list =   Mse_loss
    
    #print(f'recurrent_index :{recurrent_index}')
    min_distance = Mse_distance_list.min()

    #print(f'i :{recurrent_index} min_distance :{min_distance}')
    return min_distance

def Get_Dispersion_DataFrameIndex(original_DataFrame,original_images_x ,dispersion_images_x,DEVICE  ,calculate_distance_function=None):
  #外部輸入function提供計算距離之方法
  #需要跟images數量長度的List紀錄最短距離
  #print(f'images_x :{images_x.shape}')
  internal_DataFrame= original_DataFrame.copy()
  images_x =original_images_x.clone().float().to(DEVICE)
  internal_dispersion_images_x =dispersion_images_x.clone().float().to(DEVICE)
  distance_list= [0] * images_x.shape[0]
  #print(f'images_x.shape:{images_x.shape}')

  #1.遍歷所有的image,得到該image與其他image的距離(數量n-1 ,排除自己因為距離為0永遠是最小)
  for i, element in enumerate(images_x):
     #print(f'i : {i} , element:{element.shape}')
     if calculate_distance_function == None :
         min_distance = Calculate_Distance_Mse(i , images_x ,internal_dispersion_images_x)
     else:
         min_distance = Calculate_Distance_Dssim(i , images_x ,internal_dispersion_images_x,  calculate_distance_function)
     distance_list[i] = min_distance.item()
     #print(f'min_distance :{min_distance}')

  #將1-asb(ssim)加入dataFrame
  internal_DataFrame['Distance_dispersion'] = distance_list


  #排序 大到小
  #internal_DataFrame =internal_DataFrame.sort_values(by='DSSIM', ascending=False)

  #get largest DSSIM index
  max_value =internal_DataFrame['Distance_dispersion'].max()
  max_value_index =internal_DataFrame['Distance_dispersion'].idxmax()
  df_element = internal_DataFrame.iloc[[max_value_index]]
  
  return int(df_element['x_index'])

def Get_K_Dispersion(DataFrame ,images_x  ,k ,DEVICE, calculate_distance_function=None):
    
    original_dataFrame = DataFrame.copy()
    original_images_x =images_x.clone()
    #宣告list存放dispersion挑出的images
    dispersion_images= torch.tensor([])
    #從放k dispersion的資料
    dispersion_dataframe = pd.DataFrame(columns=original_dataFrame.columns)
    
    #get k dispersion images
    for i in range(k):
        if i == 0 :
            #第一個k取KLD最small的
            index =original_dataFrame['sum_Mse_KLD'].idxmin()
            df_extended =original_dataFrame.iloc[[index]]#  pd.Series({'KLD': 3, 'label': 4 ,'x_index': 4 , 'k': 4} )
            image_index = int(df_extended['x_index'])
            print(f'index:{index}')
            print(f'images_index:{image_index}')
            #加入k的編號,更新對應的image index 
            df_extended['k'] =i+1     
            df_extended['x_index'] =i
            
            #加入dataframe 跟 image
            dispersion_dataframe = pd.concat([dispersion_dataframe ,df_extended ])  
            dispersion_images=original_images_x[image_index].clone()
            
            #remove frim original data 
            
            original_images_x =torch.cat((original_images_x[:image_index],original_images_x[image_index+1:]))
            original_dataFrame = original_dataFrame.drop(index =index).reset_index(drop=True)
            print(f'original_dataFrame:{len(original_dataFrame)}')
            print(f'original_images_x:{original_images_x.shape}')
            
            #update images index to be the same as dataframe index 
            original_dataFrame['x_index']=original_dataFrame.index
        else:
            #選出離「已選出的點」最遠的下個點
            #Get next dataframe index 
            index = Get_Dispersion_DataFrameIndex(original_dataFrame  ,original_images_x   ,dispersion_images ,DEVICE ,calculate_distance_function)
            
            #get sub dataframe 
            df_extended =original_dataFrame.iloc[[index]]#  pd.Series({'KLD': 3, 'label': 4 ,'x_index': 4 , 'k': 4} )
            image_index = int(df_extended['x_index'])
            #print(f'index:{index}')
            #print(f'images_index:{image_index}')
            
            #加入k的編號,更新對應的image index 
            df_extended['k'] =i+1     
            df_extended['x_index'] =i
            
            #加入dataframe 跟 image
            dispersion_dataframe = pd.concat([dispersion_dataframe ,df_extended ])  
            cat_images=original_images_x[image_index].clone()
            dispersion_images =  torch.cat((dispersion_images.view(-1,784),cat_images.view(-1,784)),dim = 0)
            
            #remove frim original data
            original_images_x =torch.cat((original_images_x[:image_index],original_images_x[image_index+1:]))
            original_dataFrame = original_dataFrame.drop(index =index).reset_index(drop=True)
            #print(f'original_dataFrame:{len(original_dataFrame)}')
            #print(f'original_images_x:{original_images_x.shape}')
            
            #update images index to be the same as dataframe index 
            original_dataFrame['x_index']=original_dataFrame.index
            
    return  dispersion_dataframe , dispersion_images