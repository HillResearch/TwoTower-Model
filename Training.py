#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 09:29:23 2023

@author: haowang
"""
import torch
#import torch.nn.functional as F
# from torch.autograd import Variable
# from torch import nn
from torch.utils.data import DataLoader
# torch.manual_seed(0)

# Workspace imports
#from evaluate import evaluate_model
from Dataset import Dataset_patient
#from utils import train_one_epoch, test, plot_statistics

# Python imports
import argparse
from time import time
import numpy as np

from Embedding import TwoTowerModel

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', nargs='?', default='/Users/haowang/Desktop/hill/Data',
                        help='Input data path.')
    parser.add_argument('--data_path',nargs='?', default=['basic','diagnosis','result','treatment'],
                        help='data category.')
    parser.add_argument('--size',nargs='?',default = [0,4000],help = 'training size')
    
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--lr', type = float, default = 0.001,help = 'learning rate')
    parser.add_argument('--basic_input_dim', type=int, default = 833,help = 'input dimension of basic profile')
    
    parser.add_argument('--basic_output_dim', type=int, default = 32,help = 'output dimension of basic profile')
    
    
    parser.add_argument('--diagnosis_input_dim', type=int, default=12964,help = 'input dimension of diagnosis')
    
    parser.add_argument('--diagnosis_output_dim', type=int, default=32,help = 'output dimension of diagnosis')
    
    parser.add_argument('--treatment_input_dim', type=int, default=4,help='input dimension of treatment')
    parser.add_argument('--treatment_output_dim', type=int, default=64,help='output dimension of treatment')
    
    return parser.parse_args()
def train_one_epoch(model, data_loader, loss_fn, optimizer, epoch_no, verbose = 1):
    'trains the model for one epoch and returns the loss'
    print("Epoch = {}".format(epoch_no))
    t1 = time()
    epoch_loss = []
    # put the model in train mode before training
    model.train()
    # transfer the data to GPU
    for basic,diagnosis,treatment,output in data_loader:
        basic = basic.float()
        diagnosis = diagnosis.float()
        treatment = treatment.float()
        prediction = model(basic,diagnosis,treatment)
        output = output.float().view(prediction.size())
        loss = loss_fn(prediction, output)
        # clear the gradients
        optimizer.zero_grad()
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()
        # accumulate the loss for monitoring
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    if verbose:
        print("Epoch completed {:.1f} s".format(time() - t1))
        print("Train Loss: {}".format(epoch_loss))
    return epoch_loss

def test_one_epoch(model, data_loader, loss_fn, optimizer, epoch_no, verbose = 1):
    print("Epoch = {}".format(epoch_no))
    # Training
    # get user, item and rating data
    t1 = time()
    epoch_loss = []
    # put the model in train mode before training
    model.eval()
    # transfer the data to GPU
    for basic,diagnosis,treatment,output in data_loader:
        basic = basic.float()
        diagnosis = diagnosis.float()
        treatment = treatment.float()
        with torch.no_grad():
            prediction = model(basic,diagnosis,treatment)
            output = output.float().view(prediction.size())
        loss = loss_fn(prediction, output)
        # clear the gradients
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    if verbose:
        print("Epoch completed {:.1f} s".format(time() - t1))
        print("Test Loss: {}".format(epoch_loss))
    return epoch_loss
        





def main(args):
    print(args)

    print('create loaders...')

    Data = Dataset_patient(args.root_path,args.data_path,args.size)
    training_data = DataLoader(Data, batch_size=args.batch_size)

    
    print('create model...')
    ######
    # Model parameters
    model = TwoTowerModel(args.basic_input_dim, args.diagnosis_input_dim, 
                          args.treatment_input_dim,args.diagnosis_output_dim,
                          args.treatment_output_dim)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss()
    # best_score = np.inf
    # best_epoch, stop_round = 0, 0
    # weight_mat, dist_mat = None, None

    for epoch in range(args.n_epochs):
        print('Epoch:', epoch)
        print('training...')

        loss=train_one_epoch(model, training_data, loss_fn, optimizer, epoch_no = epoch, verbose = 1)
        print('evaluating...')
        train_loss=test_one_epoch(model, training_data, loss_fn, optimizer, epoch_no = epoch, verbose = 1)
        
        print('train_loss:', train_loss)
        #test_loss=test_one_epoch(model, test_data, loss_fn, optimizer, epoch_no = epoch, verbose = 1)
        

        # pprint('valid %.6f, test %.6f' %
        #        (val_loss_l1, test_loss_l1))

        # if val_loss < best_score:
        #     best_score = val_loss
        #     stop_round = 0
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), os.path.join(
        #         output_path, save_model_name))
        # else:
        #     stop_round += 1
        #     if stop_round >= args.early_stop:
        #         pprint('early stop')
        #         break

    # pprint('best val score:', best_score, '@', best_epoch)

    # loaders = source_loader, valid_loader, test_loader
    # loss_list, loss_l1_list, loss_r_list = inference_all(output_path, model, os.path.join(
    #     output_path, save_model_name), loaders)
    # pprint('MSE: train %.6f, valid %.6f, test %.6f' %
    #        (loss_list[0], loss_list[1], loss_list[2]))
    # pprint('L1:  train %.6f, valid %.6f, test %.6f' %
    #        (loss_l1_list[0], loss_l1_list[1], loss_l1_list[2]))
    # pprint('RMSE: train %.6f, valid %.6f, test %.6f' %
    #        (loss_r_list[0], loss_r_list[1], loss_r_list[2]))
    # pprint('Finished.')



if __name__ == '__main__':

    args = parse_args()
    
    main(args)





















