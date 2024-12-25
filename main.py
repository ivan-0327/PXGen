from torchvision import datasets
import sys
import os
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure




# Get the parent directory
current_dir  = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from Model.VAE.dataset import Process_AnchorSet, return_data
from Model.VAE.model import return_model
import argparse
import torch
from PXGen_utility.PXGen_utility import PXGen_classify_from_folder
from Utility.draw_function import draw_quadrant_chart, drew_quadrant_countBylabel, save_grid_dataframe, save_grid_img, show_quadrant_image_selectByValuesPercentage
from PXGen_utility.dispersion_algorithm import Get_K_Dispersion
from PXGen_utility.k_center import kCenter
from Model.VAE.solver import train_VAE_For_experiment
from Utility.utility import str2bool
def Execute_framwork_findQuadrant1_3(args):
    '''FrameWork'''
    #1. load model and 
    device = torch.device("cuda" if args.cuda else "cpu")
    
    args.DEVICE =device
    net = return_model(args)
    checkpoint = torch.load(os.path.join(args.load_checkpoint_dir),
                            map_location='cpu')
    
    net.load_state_dict(checkpoint)
    net = net.to(args.DEVICE)

    #2.load training data and establish boundaries and categorize
    

    #AnchorSet
    df ,x_df ,x_hat_df = PXGen_classify_from_folder(args.load_from_AnchorSet_dir,net ,args.DEVICE, args.save_g_max_values_dir)
    save_path = os.path.join(args.save_quadrantChart_dir,'test')
    drew_quadrant_countBylabel(df , 'quadrant_Mse_KLD',save_path = save_path, persentage = True)
    show_quadrant_image_selectByValuesPercentage(df,x_df ,x_hat_df,'quadrant_Mse_KLD', 50,image_save_path=save_path , select_value_type = 'Mse',ValuesPercentage =(0 ,0.05 ),changePercentToNumber= True,changeTo_KLD_limit=False)
    show_quadrant_image_selectByValuesPercentage(df,x_df ,x_hat_df,'quadrant_Mse_KLD', 15 ,image_save_path=save_path, select_value_type = 'Mse',ValuesPercentage =(0 ,0.05 ),changePercentToNumber= True,changeTo_KLD_limit=True)
    
    #3.get quadrant LILE  data and execure algorithm for find out most characteristic.
    df_LILE      = df[df['quadrant_Mse_KLD'] ==str('LILE')]
    x_df_LILE = x_df.iloc[df_LILE['x_index'].values]
    df_LILE        = df_LILE.reset_index(drop=True)
    x_df_LILE = x_df_LILE.reset_index(drop=True)
    df_LILE['x_index'] = x_df_LILE.index

    ssim = StructuralSimilarityIndexMeasure( reduction ='none')
    tensor_x = torch.tensor(x_df_LILE.values)
    
    dispersion_dataframe, dispersion_images= Get_K_Dispersion(df_LILE , tensor_x , 10 ,args.DEVICE)
    # save result
    save_path = os.path.join(args.save_quadrantChart_dir,'test')
    draw_quadrant_chart(dispersion_dataframe ,loss_type ='Mse' ,save_path = os.path.join(save_path,'Dispersion_quadrant_LILE' ))
    save_grid_img(dispersion_images ,save_path_filename =os.path.join(save_path,'Dispersion_quadrant_LILE', '{}.png'.format('Dispersion_quadrant_LILE_gird')))
    save_grid_dataframe(dispersion_dataframe,save_path_filename =os.path.join(save_path,'Dispersion_quadrant_LILE', 'quadrantChaet_table_{}.png'.format('Dispersion_quadrant_LILE')))

    #3.get quadrant HIHE  data and execure algorithm for find out most characteristic.
    df_HIHE       = df[df['quadrant_Mse_KLD'] ==str('HIHE')]
    x_df_HIHE = x_df.iloc[df_HIHE['x_index'].values]
    df_HIHE        = df_HIHE.reset_index(drop=True)
    x_df_HIHE = x_df_HIHE.reset_index(drop=True)
    df_HIHE['x_index'] = x_df_HIHE.index

    ssim = StructuralSimilarityIndexMeasure( reduction ='none')
    tensor_x = torch.tensor(x_df_HIHE.values)
    
    dispersion_dataframe, dispersion_images= Get_K_Dispersion(df_HIHE , tensor_x , 10 ,args.DEVICE)
    # save result
    save_path = os.path.join(args.save_quadrantChart_dir,'test')
    draw_quadrant_chart(dispersion_dataframe ,loss_type ='Mse' ,save_path = os.path.join(save_path,'Dispersion_quadrant_HIHE' ))
    save_grid_img(dispersion_images ,save_path_filename =os.path.join(save_path,'Dispersion_quadrant_HIHE', '{}.png'.format('Dispersion_quadrant_HIHE_gird')))
    save_grid_dataframe(dispersion_dataframe,save_path_filename =os.path.join(save_path,'Dispersion_quadrant_HIHE', 'quadrantChaet_table_{}.png'.format('Dispersion_quadrant_HIHE')))


    # K-center LILE
    df_LILE      = df[df['quadrant_Mse_KLD'] ==str('LILE')]
    x_df_LILE = x_df.iloc[df_LILE['x_index'].values]
    df_LILE        = df_LILE.reset_index(drop=True)
    x_df_LILE = x_df_LILE.reset_index(drop=True)
    df_LILE['x_index'] = x_df_LILE.index

    x = x_df_LILE.values #x為所有特徵資料
    kCenter_model = kCenter(data=x, k=10)
    cluster_category, init_centers=kCenter_model.run()
    
    centroid_indices = []
    for centroid in init_centers:
        distance = np.sum((x - centroid) ** 2, axis=1)
        centroid_index = np.argmin(distance)
        centroid_indices.append(centroid_index)
    central_points = df_LILE.iloc[centroid_indices]
    central_points_images = x_df_LILE.iloc[centroid_indices]
    central_points_images = torch.tensor(central_points_images.values)
    save_path = os.path.join(args.save_quadrantChart_dir,'test')
    draw_quadrant_chart(central_points ,loss_type ='Mse' ,save_path = os.path.join(save_path,'K-center_quadrant_LILE' ))
    save_grid_img(central_points_images ,save_path_filename =os.path.join(save_path,'K-center_quadrant_LILE', '{}.png'.format('K-center_quadrant_LILE_gird')))
    save_grid_dataframe(central_points,save_path_filename =os.path.join(save_path,'K-center_quadrant_LILE', 'quadrantChaet_table_{}.png'.format('K-center_quadrant_LILE')))
    
    # K-center HIHE
    df_HIHE       = df[df['quadrant_Mse_KLD'] ==str('HIHE')]
    x_df_HIHE = x_df.iloc[df_HIHE['x_index'].values]
    df_HIHE        = df_HIHE.reset_index(drop=True)
    x_df_HIHE = x_df_HIHE.reset_index(drop=True)
    df_HIHE['x_index'] = x_df_HIHE.index

    x = x_df_HIHE.values #x為所有特徵資料
    kCenter_model = kCenter(data=x, k=10)
    cluster_category, init_centers=kCenter_model.run()
    
    centroid_indices = []
    for centroid in init_centers:
        distance = np.sum((x - centroid) ** 2, axis=1)
        centroid_index = np.argmin(distance)
        centroid_indices.append(centroid_index)
    central_points = df_HIHE.iloc[centroid_indices]
    central_points_images = x_df_HIHE.iloc[centroid_indices]
    central_points_images = torch.tensor(central_points_images.values)
    save_path = os.path.join(args.save_quadrantChart_dir,'test')

    draw_quadrant_chart(central_points ,loss_type ='Mse' ,save_path = os.path.join(save_path,'K-center_quadrant_HIHE' ))
    save_grid_img(central_points_images ,save_path_filename =os.path.join(save_path,'K-center_quadrant_HIHE', '{}.png'.format('K-center_quadrant_HIHE_grid')))
    save_grid_dataframe(central_points,save_path_filename =os.path.join(save_path,'K-center_quadrant_HIHE', 'quadrantChaet_table_{}.png'.format('K-center_quadrant_HIHE')))

def main(args):
    Execute_framwork_findQuadrant1_3(args)
mnist_train = datasets.MNIST(
    root='./Model/VAE/content',       # 資料放置路徑
    train=True,         # 測試資料集
    download=True,       # 自動下載
)
mnist_test = datasets.MNIST(
    root='./Model/VAE/testing',       # 資料放置路徑
    train=False,         # 測試資料集
    download=True,       # 自動下載
)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VAE-TracIn self influences')
    parser.add_argument('--cuda', default=True, type=bool, help='enable cuda')
    
    # model
    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    
    
    parser.add_argument('--max_epochs', default=35, type=float, help='maximum training epochs')
    parser.add_argument('--batch_size', default=50, type=int, help='batch size')

    parser.add_argument('--latent_dim', default=200, type=int, help='dimension of the representation z')
    parser.add_argument('--x_dim', default=784, type=int, help='dimension of the input x')
    parser.add_argument('--hidden_dim', default=400, type=int, help='dimension of the hidden layer')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--train_number', default=[0], type=float, help='categories of training data')
    parser.add_argument('--train_amount', default=4000, type=int, help='training image amount per label ')
    parser.add_argument('--save_traindata_dir', default='./Model/VAE/', type=str, help='output directory')
    
    parser.add_argument('--image_size', default=28, type=int, help='image size. now only (28,28) is supported')   
    parser.add_argument('--ckpt_dir', default='./Model/VAE/checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default=None, type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--output_dir', default='./Model/VAE/outputs', type=str, help='output directory')
    
    # dataset
    
    #Our method
    parser.add_argument('--load_checkpoint_dir', default='./Model/VAE/checkpoints/Trained_model_[0].pt', type=str, help='dir of  model checkpoint.')

    parser.add_argument('--load_from_AnchorSet_dir', type=str, default='./AnchorSet/', help='dir of the training data')
    
    parser.add_argument('--save_g_max_values_dir', type=str, default='./outputs/framework', help='dir of g_max_KLD ,g_max_Mse')
    parser.add_argument('--save_quadrantChart_dir', type=str, default='./outputs/quadrantChart', help='dir of quadrantChart')
    parser.add_argument('--save_root_dir', type=str, default='./outputs/', help='dir of root')
    # output
    parser.add_argument('--n_display', type=int, default=8, help='display this number of samples with the most positive/nagetive influences')
  
    args = parser.parse_args()
    #download testing data 
    _ = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size)
    #training model
    train_VAE_For_experiment(args)

    
    #download testing data 
    _ = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size)
    args.train_amount = 0
    args.dataset_path_idx1  =os.path.join('./Model/VAE/testing' ,'MNIST','raw','t10k-labels-idx1-ubyte') 
    args.dataset_path_idx3  = os.path.join('./Model/VAE/testing'  ,'MNIST','raw','t10k-images-idx3-ubyte') 
    Process_AnchorSet(args)

    args.load_from_AnchorSet_dir = os.path.join(args.load_from_AnchorSet_dir,'data' ,'test') 
    #apply PXGen 
    main(args)