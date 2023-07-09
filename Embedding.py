#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 09:24:41 2023

@author: haowang
"""
import torch
from torch import nn

class BasicTower(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicTower, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    
class DiagnosisTower(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DiagnosisTower, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

class TreatmentTower(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TreatmentTower, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
class TwoTowerModel(nn.Module):
    def __init__(self, basic_input_dim, diagnosis_input_dim, treatment_input_dim,embedding_dim1,embedding_dim2):
        super(TwoTowerModel, self).__init__()
        self.basic_tower = BasicTower(basic_input_dim, embedding_dim1)
        self.diagnosis_tower = DiagnosisTower(diagnosis_input_dim, embedding_dim1)
        self.treatment_tower = TreatmentTower(treatment_input_dim, embedding_dim2)
        
    def forward(self, basic_input,diagnosis_input,treatment_input):
        basic_embedding = self.basic_tower(basic_input)
        diagnosis_embedding = self.diagnosis_tower(diagnosis_input)
        treatment_embedding = self.treatment_tower(treatment_input)
        
        x1 = torch.cat([basic_embedding,diagnosis_embedding], 1)
        

        x1 = x1.reshape((len(x1),64,1))



        similarity = torch.bmm(treatment_embedding,x1)
        output = torch.sigmoid(similarity)

        return output

