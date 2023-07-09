import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
class Dataset_patient(Dataset):
    def __init__(self, root_path, data_path, size):
        self.root_path = root_path
        self.data_path = data_path
        self.size = size
        #self.max_missing_ratio = max_missing_ratio
        
        
        
        self.__read_basic__(size,data_path[0])
        self.__read_diagnosis__(size,data_path[1])
    
        #self.__read_result__(size,max_missing_ratio,data_path[2])
        self.__read_treatment__(size,data_path[3])
    
    def __read_basic__(self,size,basic_profile_path):
        dfs = []
        for i in range(size[0],size[1],500):
            path = 'basic'+'_'+str(i)+'-'+str(i+500)+'.csv'
            
            df = pd.read_csv(os.path.join(self.root_path,
                                          basic_profile_path,path))
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(['Unnamed: 0'],axis = 1)
        df = df.drop(['month_of_birth/ month_of_birth_0_0'],axis = 1)
        df['year_of_birth/ year_of_birth_0_0'] = 2023-df['year_of_birth/ year_of_birth_0_0'].values
        
        df.fillna(0, inplace=True)
        matrix = df.values
        standardized_matrix = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
        self.data_basic = df.values
    
    def __read_diagnosis__(self,size,diagnosis_path):
        dfs = []
        for i in range(size[0],size[1],500):
            path = 'diagnosis'+'_'+str(i)+'-'+str(i+500)+'.csv'
            
            df = pd.read_csv(os.path.join(self.root_path,
                                          diagnosis_path,path))
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(['Unnamed: 0'],axis = 1)

        
        df.fillna(0, inplace=True)
        non_float_cols = df.columns[df.dtypes != 'float']
        df = df.drop(columns=non_float_cols)
        matrix = df.values
        standardized_matrix = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
        
        self.data_diagnosis = df.values
    
    # def __read_result__(self,size,max_missing_ratio,result_path):
    #     dfs = []
    #     for i in range(size[0],size[1],500):
            

    #         path = 'result'+'_'+str(i)+'-'+str(i+500)+'.csv'
    #         print(path)
            
    #         df = pd.read_csv(os.path.join(self.root_path,
    #                                       result_path,path))
    #         dfs.append(df)
    #     df = pd.concat(dfs, ignore_index=True)
    #     missing_ratios = df.isnull().sum() / len(df)
    #     cols_to_drop = missing_ratios[missing_ratios > max_missing_ratio].index
    #     df_clean = df.drop(cols_to_drop, axis=1)
    #     self.data_result = df_clean.values
    def __read_treatment__(self,size,treatment_path):
        dfs = []
        for i in range(size[0],size[1],500):
            path = 'treatment'+'_'+str(i)+'-'+str(i+500)+'.csv'
            
            df = pd.read_csv(os.path.join(self.root_path,
                                          treatment_path,path))
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(['Unnamed: 0'],axis = 1)
        operation_name = []
        for i in range(2,3000,5):
            operation_name.append(df.columns[i])
        Operation = []
        for i in operation_name:
            df1 = df[i]
            Operation.append(df1.apply(lambda x: 0 if pd.isna(x) else 1))
        Operation_output = pd.concat(Operation, axis = 1)
        Operation_output.columns = operation_name
        operation_feature = []
        for i in range(3000):
            if df.columns[i] not in operation_name:
                operation_feature.append(df.columns[i])
        Operation_feature = df[operation_feature]
        Operation_feature.fillna(-1099.0, inplace=True)
        I = []
        for i,name in enumerate(Operation_feature.dtypes):
            if name== 'int64':
                I.append(i)
        Operation_feature[Operation_feature.columns[I]] = Operation_feature[Operation_feature.columns[I]].astype('float64')
        
        Operation_feature[Operation_feature.columns[1901]] = 0.0
        Operation_feature[Operation_feature.columns[1905]] = 0.0
        Operation_feature[Operation_feature.columns[1909]] = 0.0
        matrix = Operation_feature.values
        standardized_matrix = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
        
        self.Operation_feature = Operation_feature.values.reshape((len(standardized_matrix),600,4))
        self.Operation_output = Operation_output.values
        
        
        
        
    def __len__(self):
        return len(self.data_basic)
    def __getitem__(self, index):
        'Generates one sample of data'

        # get the train data
        basic = self.data_basic[index]
        diagnosis = self.data_diagnosis[index]
        #result = self.data_result[index]
        treatment = self.Operation_feature[index]
        output = self.Operation_output[index]

        return basic,diagnosis,treatment,output
        
        
        


# root_path= '/Users/haowang/Desktop/hill/Data' 
# data_path=['basic','diagnosis','result','treatment']
# size = [0,1000]

# Data = Dataset_patient(root_path,data_path,size)

# training_data = DataLoader(Data, batch_size=10)

# for x,y,z,g in training_data:
#     print(z.shape)
        

        
