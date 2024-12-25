'''solver.py'''

import math
import os
import sys
# Get the parent directory
current_dir  = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from  Model.VAE.model import return_model ,loss_function
from Model.VAE.utils import print_size
from Model.VAE.dataset import return_data, return_data_experiment
from torchvision.utils import make_grid, save_image
class Solver(object):
    def __init__(self, args):
        self.args =args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.DEVICE = torch.device("cuda" if self.use_cuda else "cpu")
        self.max_epochs = args.max_epochs
        self.global_iter = 0

        self.latent_dim = args.latent_dim
        self.x_dim = args.x_dim
        self.hidden_dim = args.hidden_dim
        self.lr = args.lr
        self.batch_size = args.batch_size

        self.has_label = False  # does the dataset has labels?

        

        

        self.net = return_model(self)
        print_size(self.net)
        self.optim = Adam(self.net.parameters(), lr=self.lr)
        
        # checkpoint dir
        self.ckpt_dir = os.path.join(args.ckpt_dir)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name 
        # load the checkpoint if the check dir is not None
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name , self.net)
        
        #output dir 
        
        self.output_dir = os.path.join(args.output_dir)
        self.output_dir_input_reconstruct = os.path.join(args.output_dir,"input_reconstruct")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.output_dir_input_reconstruct):
            os.makedirs(self.output_dir_input_reconstruct, exist_ok=True)

        #set checkpoint amount
        
        # get dataLoader
        self.train_loader = return_data(args)

    def  use_return_data_experiment(self):
        train_loader , training_images , experiement_images , training_images_labels ,experiment_images_labels  = return_data_experiment(self.args) 

        self.train_loader =train_loader
        self.training_images =training_images
        self.experiement_images =experiement_images
        self.training_images_labels =training_images_labels
        self.experiment_images_labels =experiment_images_labels

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def viz_reconstruction(self , x_tensor , x_recon_tensor ,iteration):
        self.net_mode(train=False)
        x = x_tensor.cpu()
        x = make_grid(x.view(-1, 1, 28, 28), nrow=10, normalize=True)
        
        x_recon = x_recon_tensor.cpu()
        x_recon = make_grid(x_recon.view(-1, 1, 28, 28), nrow=10, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        
        self.net_mode(train=True)
        save_image(tensor=images,
                               fp=os.path.join(self.output_dir_input_reconstruct, '{}.jpg'.format(iteration)),
                               nrow=10, pad_value=1)

    def save_checkpoint(self, filename, silent=True):

        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        
        states = {'iter':self.global_iter,
                  
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename , net):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location=self.DEVICE )
            self.global_iter = checkpoint['iter']
            
            net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))  

    def generate_to_dir(self , path):
        net = return_model(self)
        self.load_checkpoint(self.ckpt_name , net)
        generated_images =net.generate_batch_images(3500)
        for i in range(generated_images.shape[0]):
            images = generated_images[i].view(1,28,28)

            save_path=os.path.join(path,str(i)+'.png')
            save_image(images,save_path)

    def train(self):
        self.net_mode(train=True)
        dataset_size = len(self.train_loader)
       
        total_iter =  self.max_epochs*dataset_size
        save_iter  =  int(total_iter/ self.save_checkpoint_amount  )      

        pbar = tqdm(total=total_iter)
        pbar.update(self.global_iter)

        for epoch in range(self.max_epochs):
            overall_loss = 0
            for batch_idx, x in enumerate(self.train_loader):
                self.global_iter += 1
                pbar.update(1)
            
                x = x.to(self.DEVICE)
            
                self.optim.zero_grad()

                x_hat, mean, log_var = self.net(x)
                loss = loss_function(x, x_hat, mean, log_var)

                overall_loss += loss.item()

                loss.backward()
                self.optim.step()

                
                
                if self.global_iter%save_iter == 0:
                    pbar.write('Average Loss :{:.3f} , iteration :{:.3f} '.format(
                        overall_loss / (batch_idx *self.batch_size) , self.global_iter ))
                    self.save_checkpoint(str(self.global_iter))
                    self.viz_reconstruction(x , x_hat ,self.global_iter)
            
            #print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
        self.save_checkpoint(str(self.global_iter))
        pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
        print("Finish!!")
        pbar.write("[Training Finished]")
        pbar.close()
    def train_no_save_and_returnModel(self):
        self.net_mode(train=True)
        dataset_size = len(self.train_loader)
        total_iter =   self.max_epochs*dataset_size
        
        
        pbar = tqdm(total=total_iter)
        pbar.update(self.global_iter)
        for epoch in range(self.max_epochs):
            overall_loss = 0
            for batch_idx, x in enumerate(self.train_loader):
                self.global_iter += 1
                pbar.update(1)
            
                x = x.to(self.DEVICE)
            
                self.optim.zero_grad()

                x_hat, mean, log_var = self.net(x)
                loss = loss_function(x, x_hat, mean, log_var)

                overall_loss += loss.item()

                loss.backward()
                self.optim.step()
                
            
            #print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
        pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
        print("Finish!!")
        pbar.write("[Training Finished]")
        pbar.close()
        
        return self.net
        

def train_VAE_For_experiment(args, pre_fix='Trained_model_'):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    args.DEVICE = torch.device("cuda" if use_cuda else "cpu")
    net = Solver(args)

    if args.train:
        model =net.train_no_save_and_returnModel()
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir, exist_ok=True)
        file_path = os.path.join(args.ckpt_dir, f'{pre_fix}{args.train_number}.pt')
        torch.save(model.state_dict(), file_path)


def train_VAE_For_experiment1(args, pre_fix='Trained_model_'):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    args.DEVICE = torch.device("cuda" if use_cuda else "cpu")
    net = Solver(args)
    net.use_return_data_experiment()

    if args.train:
        model =net.train_no_save_and_returnModel()
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir, exist_ok=True)
        file_path = os.path.join(args.ckpt_dir, f'{pre_fix}{args.train_number}.pt')
        torch.save(model.state_dict(), file_path)
    return net.training_images , net.experiement_images , net.training_images_labels ,net.experiment_images_labels

        