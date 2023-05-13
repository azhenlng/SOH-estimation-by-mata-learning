import os
import time
import sys
import json
import argparse
import torch
import random
import warnings
import copy
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
from model.arch import gcn, rnn
from preprocess.dataset import SlidingWindowBattery, PreprocessNormalizer
from preprocess.pre_utils import Normalizer, LabelNormalizer, PredictResult, EDNormalizer, EDPredictResult
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import random
import optuna

import logging

class Train:
    """
    for training
    """

    def __init__(self, args):
        """
        initialization, load project arguments
        """
        self.args = args
        self.current_epoch = 1
        self.step = 0
        self.train_step = 1
        self.test_step = 1
        self.loss_dict = OrderedDict()
        self.task_type = args.task_type

        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        current_path = os.path.join(self.args.save_model_path, time_now)
        self.mkdir(current_path)
        self.current_path = current_path
        current_model_path = os.path.join(current_path, "model")
        loss_picture_path = os.path.join(current_path, "loss")
        current_result_path = os.path.join(current_path, "result")
        current_tb_path = os.path.join(current_path, "tb")
        # create folders
        self.mkdir(current_model_path)
        self.mkdir(loss_picture_path)
        self.mkdir(current_result_path)
        self.mkdir(current_tb_path)
        
        self.args.current_path = current_path
        self.args.current_model_path = current_model_path
        self.args.loss_picture_path = loss_picture_path
        self.args.current_result_path = current_result_path
        self.args.current_tb_path = current_tb_path
        self.writer = SummaryWriter(current_tb_path)
        self.normalizer = None
        self.model = None        

        self.dataloader_maml=[]
        self.len_train_maml=[]

    @staticmethod
    def mkdir(path):
        """
        create folders
        :param path: path
        """
        if os.path.exists(path):
            print('%s is exist' % path)
        else:
            os.makedirs(path)


    def main(self,optuna_params={}):
        """
        training
        load training data, preprocessing, create & train & save model, save parameters
        train_pre: normalized data
        model: model
        loss: nll kl label
        rec_error: reconstruct error
        """

        start_time = time.time()
        params = dict(
            device=self.args.device,
            embed_dim=self.args.window_len,
            class_num=self.args.class_num,
            kernel_num=self.args.kernel_num,
            kernel_sizes=self.args.kernel_sizes,
            drop_out=self.args.drop_out,
            last_kernel_num=self.args.last_kernel_num,
            task_type=self.task_type
        )
        self.args.batch_size={"NE":2048,"TsH":1024,"NCA":512,"NCM":512,"NCM_NCA":256}
       
        for i in self.args.data_path.keys():
            self.train_file = []
            self.test_file = []
            self.data_list = sorted(os.listdir(self.args.data_path[i]))
            self.split_train_test_cell()
            self.args.interpolate = 1
            self.args.is_interpolate = True
            if i == "TsH":
                max_num = 50
            if i == "NE":
                max_num = 1.1
            if i == "NCA" or i =="NCM":
                max_num = 3500
            if i == "NCM_NCA":
                max_num = 2510
            data_train = SlidingWindowBattery(data_path=self.args.data_path[i],
                                          file_list=self.train_file,
                                          exp_data=self.args.exp_data,
                                          window_len=self.args.window_len,
                                          interval=self.args.interval,
                                          jobs=self.args.jobs,
                                          ram=self.args.ram,
                                          sigvolt_nums=self.args.sigvolt_nums,
                                          window_limit=self.args.window_limit,
                                          interpolate=self.args.interpolate,
                                          partition=self.args.partition,
                                          partition_num=self.args.partition_num,
                                          is_interpolate=self.args.is_interpolate
                                          )

        
            self.label_normalizer = EDNormalizer().exp_minmaxscaler(min_num=0, max_num=max_num)
            self.normalizer = Normalizer(dfs=[data_train[i][0] for i in range(20)], )
            train_pre = PreprocessNormalizer(data_train, norm_name=self.args.norm, normalizer_fn=self.normalizer.norm_func)

        # DataLoader
            train_loader = DataLoader(dataset=train_pre, batch_size=self.args.batch_size[i], shuffle=False,num_workers=0, drop_last=True, )
            self.dataloader_maml.append(train_loader)
            self.len_train_maml.append(len(train_pre))
        print(f'Dataset complete in {time.time() - start_time}s')
        min_index = np.argmin(self.len_train_maml)
        self.iteration_min = int(self.len_train_maml[min_index] / min(self.args.batch_size.values()))
            
        # specify model
        if self.args.model_type == 'gcn':
            
            model = to_var(gcn.CNN_Gate_Aspect_Text(**params), self.args.device).float()
        else:
            model = None
        print("model", model)
        
         # specify optimizer and learning scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs,
                                      eta_min=self.args.cosine_factor * self.args.learning_rate)
        if self.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.SmoothL1Loss()

        
        test_best_rmse=float('inf')
        p_bar = tqdm(total=self.iteration_min * self.args.epochs, 
                                 desc='training', 
                                 ncols=160, mininterval=1,
                                 maxinterval=10, miniters=1)
        while self.current_epoch <= self.args.epochs:
            total_loss=0
            dataloader_iter_list = [iter(dl) for dl in self.dataloader_maml]
            for iteration in range(self.iteration_min):
                self.step += 1
                try:
                    batch_lst = [next(dl) for dl in dataloader_iter_list]
                except StopIteration:
                    pass
                grads,loss = self.MAML(model=model, batch_lst = batch_lst)
                
                model.train()
                # update parameters
                optimizer.zero_grad()
                for index, (name , params) in enumerate(model.named_parameters()):
                    params.grad = grads[index]
                
                optimizer.step()

                total_loss += loss.item()
                loss_info = {'mean_loss': total_loss / (1 + iteration)}
                
                self.loss_dict[f"epoch{self.current_epoch}"] = {'epoch_train_loss': total_loss / (1 + iteration)}                                         
                p_bar.set_postfix(loss_info)
                p_bar.set_description('training - Epoch %d/%i' % (self.current_epoch,self.args.epochs))
                
                # save model
               
                p_bar.update(1)      
            self.current_epoch += 1
        p_bar.close()
        self.model_result_save(model)
        
        return self.args 

    def MAML(self, model, batch_lst):
        """
       maml_training

        load training data, maml preprocessing, calculate grads
        param model: model
        param batch_lst: batch in train data
        """
        running_loss, running_error, iteration, data_len, running_rmse = 0, 0, 0, 0, 0
        grads_lst=[]
        #create inner_model
        for idx, (input_data, metadata) in enumerate(batch_lst):
            inner_model = copy.deepcopy(model)
            outer_model_pm = OrderedDict(model.named_parameters())
            inner_model.load_state_dict(outer_model_pm)
            inner_model_pm = OrderedDict(inner_model.named_parameters())            
            input_data_in = input_data[::2]
            input_data_out = input_data[1::2]            
            metadata_in = copy.deepcopy(metadata)
            metadata_out = copy.deepcopy(metadata)
            for k in metadata.keys():
                metadata_in[k] = metadata_in[k][::2]
                metadata_out[k] = metadata_out[k][1::2]
                             
            self.model_grad_ini(model, input_data_in, metadata_in, outer_model_pm)  ###model grad initialized      
            self.inner_train(model, input_data_in, metadata_in, inner_model_pm) ###inner train loop 
            loss = self.loss_cal(inner_model,input_data_out, metadata_out)
            grads = torch.autograd.grad(loss, inner_model.parameters(), create_graph = True)
            grads_lst.append(grads)
        grads_mean = tuple(np.array(grads_lst).mean(axis=0))
        return grads_mean, loss    


    def loss_cal(self,model_cal,input_data,metadata):
        """
       calculate loss

        param model_cal: model used for calculating
        param input_data: transformed data
        param metadata: target
        """
        outputs = model_cal(to_var(input_data, self.args.device))
        labels = self.label_normalizer.transform(metadata['label'])
        labels = torch.tensor(labels).to(torch.float32)
        labels = to_var(labels, self.args.device)
        criterion = nn.SmoothL1Loss(reduction='mean')
        loss = criterion(outputs, labels)
        
        return loss
  
    def model_grad_ini(self,model, input_data, metadata, outer_model_pm):
        """
       initialize outer_model grads

        param model: outer model

        """
        for name , params in model.named_parameters():
            if params.grad == None: 
                ini_loss = self.loss_cal(model, input_data, metadata)
                ini_loss.backward()
                model.load_state_dict(outer_model_pm)
                print("model grad initialization completed")
                
    def inner_train(self,inner_model, input_data, metadata, inner_model_pm):
        """
       inner train loop
        """
        for inner_step in range(self.args.inner_step):
            inner_loss = self.loss_cal(inner_model, input_data, metadata)
            grads = torch.autograd.grad(inner_loss, inner_model.parameters(), create_graph = True)
            inner_model_pm = OrderedDict((name, param - self.args.inner_lr * grad)
                                         for ((name, param), grad) in zip(inner_model_pm.items(), grads))
            inner_model.load_state_dict(inner_model_pm)
               
    def split_train_test_cell(self):
        """
       split train/test file according to ratio
        """
        file_name_set = set()
        for file in self.data_list:
            file_list = os.path.splitext(file)[0].split('_')
            file_name = file_list[0]+'_'+file_list[1]
            file_name_set.add(file_name)
        train_file = random.sample(list(file_name_set), int(len(file_name_set)*self.args.train_test_split_ratio))
        test_file = list(set(file_name_set).difference(set(train_file)))
        for file in self.data_list:
            file_list = os.path.splitext(file)[0].split('_')
            file_name = file_list[0] + '_' + file_list[1]
            if file_name in train_file:
                self.train_file.append(file)
            elif file_name in test_file:
                self.test_file.append(file)

        random.shuffle(self.train_file)
        random.shuffle(self.test_file)

   

    def model_result_save(self, model):
        """
        save model
        :param model: vae or transformer
        """
        model_params = {'train_time_start': self.current_path,
                        'train_time_end': time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                        'args': vars(self.args),
                        'loss': self.loss_dict
                        }

        json.dump(model_params, open(os.path.join(self.args.current_model_path, 'model_params.json'), 'w'),
                  indent=4)

        torch.save(model.state_dict(), os.path.join(self.args.current_model_path, "model.torch"))
        json.dump({k: list(v) for k, v in self.normalizer.__dict__.items()},
                  open(os.path.join(self.args.current_model_path, "norm.json"), 'w'), indent=4)

def to_var(x, device='cpu'):
    """
    put data into cuda
    :param x: data or model
    :param device cpu / gpu
    """
    if device == 'cuda':
        x = x.cuda()
    return x


def main(args):
    """
    train maml base model

    """
    for index, learning_rate in enumerate([0.005,0.0001]):
        args.learning_rate = learning_rate
        for index, cosine_factor in enumerate([0.6,0.9]):
            args.cosine_factor = cosine_factor
            for index, drop_out in enumerate([0.3,0.5]):
                args.drop_out = drop_out
                for index, last_kernel_num in enumerate([7,16]):
                    args.last_kernel_num = last_kernel_num
                    for index, inner_step in enumerate([3,10]):
                        args.inner_step = inner_step
                        Train(args).main()



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Automatic parameter setting')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(os.getcwd()), 'code/maml_param.json'))
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)
    main(args)


