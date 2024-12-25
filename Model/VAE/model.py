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
import os 



    
'''Define our model: Variational AutoEncoder (VAE)'''

"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim =latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):

        h_       = self.LeakyReLU(self.FC_input(x))

        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance
                                                       #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.latent_dim =latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder , DEVICE):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.DEVICE  = DEVICE
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.DEVICE)        # sampling epsilon
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def encode(self , x):
        
        mean, log_var = self.Encoder(x)    
        
        return mean, log_var
    
    def decode(self , z):
        x_hat            = self.Decoder(z)
        
        return x_hat
    def generate_batch_images(self,batch_size ):
        with torch.no_grad():
            noise = torch.randn(batch_size, self.Decoder.latent_dim).to(self.DEVICE)
            generated_images = self.decode(noise)
        return generated_images #(generated_images*255)

    def forward(self, x):
        #print(f'x shape :{x.shape}')
        mean, log_var = self.Encoder(x)

        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)

        return x_hat, mean, log_var
    
    '''Define Loss function (reprod. loss) and optimizer'''
    from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def loss_function_separate(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='none')

    KLD               = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp(),dim =1)
    reproduction_loss = torch.sum(reproduction_loss,dim =1)
    return reproduction_loss ,KLD

def loss_functionKLD( mean, log_var):


    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    '''將各圖片的KL分開算'''
    KLDMatrix = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp(),dim =1)
    
    return  KLD ,KLDMatrix



def return_model(args):
    encoder = Encoder(input_dim=args.x_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    decoder = Decoder(latent_dim=args.latent_dim, hidden_dim = args.hidden_dim, output_dim = args.x_dim)

    model = Model(Encoder=encoder, Decoder=decoder ,DEVICE = args.DEVICE).to(args.DEVICE)

    return model 