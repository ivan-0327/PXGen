import pandas as pd
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F

class Criterion_Calculater:
    def __init__(self , model ,loader, label_dic , KLD_function  ,DEVICE ):
        self.model        = model
        self.loader        = loader
        self.KLD_function = KLD_function
        self.SSIM_function = StructuralSimilarityIndexMeasure( reduction ='none')
        self.DEVICE        = DEVICE
        self.label_dic     = label_dic

        #self.mes_loss_function =
    def __Get_Mse__(self,x ,x_hat):
        Mse_loss = F.mse_loss(x_hat.view(-1,784).cpu(), x.view(-1,784).cpu(), reduction='none')
        #print(f'Mse_loss :{Mse_loss.shape} ')                       
        Mse_loss = torch.sum(Mse_loss,dim =1)  
        #print(f'Mse_loss :{Mse_loss.shape} ')   
        return Mse_loss
    def __Get_KLD_Error__(self,x , x_hat ,mean , log_var ):
        
        _ ,KLD   =self.KLD_function( mean.cpu(), log_var.cpu())
        
        return KLD
    def __Get_DSSIM__(self,x , x_hat  ):
        #print(f'x :{x.shape} ')  
        DSSIM= (1-self.SSIM_function(x.view(-1,1,28,28), x_hat.view(-1,1,28,28)) )/2
        #print(f'DSSIM :{DSSIM.shape} ')   
        return DSSIM
        
    def __Getlabel_from_dic__(self , labels):
        '''一個一個處理'''
        map_lable = []
        for lable in labels:
            new =self.label_dic[int(lable)]  
            map_lable.append(new)
            
        return map_lable
    def Get_KLD_And_MSE_DSSIM(self ):
        df   = pd.DataFrame({'Mse' : [] , 'KLD':[] , 'Dssim':[] , 'label':[] , 'x_index':[] ,'quadrant_Mse_KLD' :[] ,'quadrant_Dssim_KLD' :[] })
        x_df = pd.DataFrame()
        x_hat_df = pd.DataFrame()
        '''從loader中取得batch data 處理'''
        for x ,label in self.loader:
            '''使用 pandas處理資料'''
            df_empty = pd.DataFrame({'Mse' : [] , 'KLD':[] , 'Dssim':[] , 'label':[] , 'x_index':[],'quadrant_Mse_KLD' :[] ,'quadrant_Dssim_KLD' :[] })
            '''使用模型推理'''
            x_hat, mean, log_var= self.model(x.view(-1,784).to(self.DEVICE))
             
            #print(f'mean:{mean.shape} , log_var:{log_var.shape}')
            '''建立 x dataframe'''
            x_df_empty = pd.DataFrame(x.view(-1,784).numpy()).reset_index(drop =True)
            x_hat_df_empty = pd.DataFrame(x_hat.detach().cpu().view(-1,784).numpy()).reset_index(drop =True)
            '''計算Loss'''
            KLD =self.__Get_KLD_Error__(x, x_hat, mean, log_var )
            #print(f'KLD :{KLD.shape} ')
            '''計算Mse'''
            Mse_loss =self.__Get_Mse__(x, x_hat )
            #print(f'Mse_loss :{Mse_loss.shape} ')
            '''計算DSSIM'''
            DSSIM = self.__Get_DSSIM__(x.to(self.DEVICE), x_hat.to(self.DEVICE) )
            
            #print(f'x_df :{x_df.shape} ')
            df_empty['Mse']              = Mse_loss.detach().cpu().numpy()
            df_empty['KLD']              = KLD.detach().cpu().numpy()
            df_empty['Dssim']            = DSSIM.detach().cpu().numpy()
            df_empty['label']            = self.__Getlabel_from_dic__(label)
            
            
            
            df       = pd.concat([df, df_empty])
            x_df     = pd.concat([x_df, x_df_empty])
            x_hat_df = pd.concat([x_hat_df, x_hat_df_empty])
            
        x_df = x_df.reset_index(drop =True)
        x_hat_df = x_hat_df.reset_index(drop =True)
        '''give df x_index'''
        df['x_index']   = x_df.index
        df = df.reset_index(drop =True)
        return df ,x_df ,x_hat_df
 