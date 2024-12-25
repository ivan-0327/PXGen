
import math
import os
from matplotlib import pyplot as plt
import torch
from torch.optim import Adam
from tqdm import tqdm
from model import return_model ,loss_function
from utils import print_size
from dataset import return_data
from torchvision.utils import make_grid, save_image

def load_checkpoint( a ,filename):
        file_path = os.path.join('checkpoints', filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location=a.DEVICE )
            global_iter = checkpoint['iter']
            
            a.net.load_state_dict(checkpoint['model_states']['net'])
            #a.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))  

def show_image(x, idx):
    x = x.view(x.shape[0], 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy(), cmap='gray')
             
class A:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.DEVICE = torch.device("cuda" if self.use_cuda else "cpu")
        self.max_epochs = 50
        self.global_iter = 0

        self.latent_dim = 200
        self.x_dim = 784
        self.hidden_dim = 400

        self.lr = 1e-3    
        self.net =None

if __name__ == "__main__":
        
    a= A()    

    a.net = return_model(a)
    load_checkpoint(a , 'last')

    generated_images =a.net.generate_batch_images(3500)
    for i in range(generated_images.shape[0]):
        images = generated_images[i].view(1,28,28)

        save_path='./outputs/generation'+'/'+str(i)+'.png'
        save_image(images,save_path)
        #show_image(generated_images, idx=i)
    
    #grid = make_grid(generated_images.view(-1, 1, 28, 28), nrow=10, normalize=True)
    #images = torch.stack([grid, grid], dim=0).cpu()
    #save_image(tensor=images,
    #                           fp=os.path.join('outputs', '{}.jpg'.format('test')),
    #                           nrow=10, pad_value=1)
    
        
        