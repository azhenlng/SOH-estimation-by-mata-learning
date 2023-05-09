import os
import time
import sys
import json
import argparse
import torch
import random
import warnings

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
        self.step = 1
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
        self.data_list = sorted(os.listdir(self.args.data_path))
        self.train_file = []
        self.test_file = []


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
        self.split_train_test_cell()
        if self.args.optuna:
            self.args.learning_rate = optuna_params['learning_rate']
            self.args.epochs = optuna_params['epochs']
            self.args.batch_size = optuna_params['batch_size']
            self.args.cosine_factor = optuna_params['cosine_factor']
            self.args.drop_out = optuna_params['drop_out']
            self.args.last_kernel_num = optuna_params['last_kernel_num']

        self.args.is_interpolate = True
        self.args.interpolate = 1
        if "Tsh" in self.args.data_path:
            max_num = 50
        if "NE" in self.args.data_path:
            max_num = 1.1
        if "NCA" in self.args.data_path or "NCM" in self.args.data_path:
            max_num = 3500
        if "NCM_NCA" in self.args.data_path:
            max_num = 2510

        
        data_train = SlidingWindowBattery(data_path=self.args.data_path,
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
        data_test = SlidingWindowBattery(data_path=self.args.data_path,
                                         file_list=self.test_file,
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
        
        test_pre = PreprocessNormalizer(data_test, norm_name=self.args.norm, normalizer_fn=self.normalizer.norm_func)
        print(f'Dataset complete in {time.time() - start_time}s')

        if self.args.partition:
            numlst_train = data_train.getnumlst()
            numlst_test = data_test.getnumlst()
        # DataLoader
            train_loader = DataLoader(dataset=train_pre, batch_size=self.args.batch_size,
                                      sampler=PartSampler(numlst_train),
                                      num_workers=self.args.jobs,
                                      drop_last=False,
                                      pin_memory=True if self.args.device == 'cuda' else False
                                      )
            test_loader = DataLoader(dataset=test_pre, batch_size=self.args.batch_size,
                                     sampler=PartSampler(numlst_test),
                                     num_workers=self.args.jobs,
                                     drop_last=False,
                                     pin_memory=True if self.args.device == 'cuda' else False
                                     )
        else:
            train_loader = DataLoader(dataset=train_pre, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=0, drop_last=True, )
            test_loader = DataLoader(dataset=test_pre, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=0, drop_last=True, )

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
        
        
        # specify model
        if self.args.model_type == 'gcn':
            self.model = to_var(gcn.CNN_Gate_Aspect_Text(**params), self.args.device).float() 
        
        self.writer.add_graph(self.model, to_var(torch.zeros(self.args.batch_size, self.args.window_len, data_test[0][0].shape[1]), self.args.device))
        print("model", self.model)
    
     # specify optimizer and learning scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs,
                                      eta_min=self.args.cosine_factor * self.args.learning_rate)
        if self.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.SmoothL1Loss()

        
        test_best_rmse=float('inf')

        while self.current_epoch <= self.args.epochs:
            print(f'Epoch{self.current_epoch}:')
            # calculate loss
            train_running_loss, train_rmse, train_mape_error, train_predict_result = self.running_program(mode = 'train',
                                                                            data_loader = train_loader,
                                                                            criterion =criterion, optimizer = optimizer,
                                                                            scheduler= scheduler)

            test_running_loss, test_rmse, test_mape_error, test_predict_result = self.running_program(mode = 'test',
                                                                          data_loader = test_loader,
                                                                          criterion = criterion)
            epoch_train_loss = train_running_loss
            epoch_test_loss = test_running_loss
           

            print(f"epoch_train_loss {epoch_train_loss}")
            print(f"epoch_test_loss {epoch_test_loss}")
            print(f"train_rmse {train_rmse}")
            print(f"test_rmse {test_rmse}")
          
            print(f"train_error {train_mape_error}")
            print(f"test_error {test_mape_error}")
            self.loss_dict[f"epoch{self.current_epoch}"] = {'epoch_train_loss': epoch_train_loss,
                                                            "epoch_test_loss": epoch_test_loss,
                                                            
                                                            "train_rmse": train_rmse,
                                                            "test_rmse": test_rmse,
                                                            "train_error": train_mape_error,
                                                            "test_error": test_mape_error}
            if test_rmse<=test_best_rmse:
                self.model_result_save(self.model)
                test_best_rmse=test_rmse
                self.loss_visual()

           
            self.current_epoch += 1

        return self.args,  test_best_rmse

    def running_program(self, mode,  data_loader, criterion, optimizer=None, scheduler=None):
        """
        training process
        :param mode: train,test
        :param data_loader: train data_loader, test data_loader
        :param criterion: loss func
        :param optimizer: 
        :param scheduler: 
        :return: loss
        """
        predict_result = EDPredictResult(cluster_nums=self.args.cluster_nums, sigvolt_nums=self.args.sigvolt_nums)
        running_loss, running_error, iteration, data_len, running_rmse = 0, 0, 0, 0, 0
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        for idx, (input_data, metadata) in enumerate(tqdm(data_loader)):
            outputs = self.model(to_var(input_data, self.args.device))
            if self.task_type == "classification":
                labels = metadata['label'].to(torch.float32)
                labels = to_var(labels, self.args.device)
                labels = labels.long()
            else:
                labels = self.label_normalizer.transform(metadata['label'])
                labels = torch.tensor(labels).to(torch.float32)
                labels = to_var(labels, self.args.device)

            restored_outputs = self.label_normalizer.inverse_transform(outputs.detach().cpu().numpy())

            loss = criterion(outputs, labels)

            error = self.calculate_error(restored_outputs, labels)
            lens = len(labels)
            running_error += error
            data_len += lens
            
            # update parameters
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            rmse = np.sqrt(mean_squared_error(outputs.detach().cpu(), labels.cpu()))
            running_rmse += rmse
            running_rmse_epoch = running_rmse / (1 + iteration)


            running_loss += loss.item()
            running_loss_epoch = running_loss / (1 + iteration)
            loss_info = {'running_loss': running_loss_epoch}
            self.tensorboard_loss(loss_info, mode)

            self.step += 1
            if mode == 'train':
                self.train_step += 1
            else:
                self.test_step += 1
            iteration += 1
       
        mape_error = running_error/data_len

        predict_result.calculate()

        return running_loss_epoch, running_rmse_epoch, mape_error, predict_result

    def split_train_test_cell(self):
        """
       split train/test file according to train_num
        """
        self.train_file=random.sample(self.data_list, self.args.train_num)
        self.test_file = list(set(self.data_list).difference(set(self.train_file)))
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

    def calculate_error(self, output, label):
        label = label.cpu().numpy().reshape((len(label), 1))
        restored_label = self.label_normalizer.inverse_transform(label)
        error = abs(restored_label - output) / restored_label
        return np.sum(error)

    def tensorboard_loss(self, loss_info, mode):
        """
        save iteration loss in tensorboard
        """
        self.writer.add_scalar('running_loss', loss_info['running_loss'], self.step)
        if mode == 'train':
            self.writer.add_scalar('train_loss', loss_info['running_loss'], self.train_step)
        else:
            self.writer.add_scalar('test_loss', loss_info['running_loss'], self.test_step)

   
    def loss_visual(self):
        """
        draw loss curve
        """
        if self.args.epochs == 0:
            return
        x = list(self.loss_dict.keys())
        df_loss = pd.DataFrame(dict(self.loss_dict)).T.sort_index()
        epoch_train_loss = df_loss['epoch_train_loss'].values.astype(float)
        epoch_test_loss = df_loss['epoch_test_loss'].values.astype(float)
       
        train_mape_error = df_loss['train_error'].values.astype(float)
        test_mape_error = df_loss['test_error'].values.astype(float)


        plt.figure()
        plt.subplot(2, 3, 1)
        plt.plot(x, epoch_train_loss, 'bo-', label='epoch_train_loss')
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(x, epoch_test_loss, 'bo-', label='epoch_test_loss')
        plt.legend()

       

        plt.subplot(2, 3, 5)
        plt.plot(x, train_mape_error, 'bo-', label='train_mape_error')
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(x, test_mape_error, 'bo-', label='test_mape_error')
        plt.legend()

        plt.savefig(self.args.loss_picture_path + '/' + 'loss.png')
        plt.close('all')

def to_var(x, device='cpu'):
    """
    put data into cuda
    :param x: data or model
    :param device cpu / gpu
    """
    if device == 'cuda':
        x = x.cuda()
    return x


class PartSampler(Sampler):
    """
    for partition dataloader 

    """

    def __init__(self, lst):
        self.lst = [i for i in lst]
        self.a = []

        sum = 0
        for i in range(len(self.lst)):
            for index in random.sample(range(sum, sum + self.lst[i]), self.lst[i]):
                self.a.append(index)
            sum += self.lst[i]

    def __iter__(self):
        return iter(self.a)

    def __len__(self):
        return len(self.a)


def train_optuna(trial):
    """
    train with optuna

    """     
    params_train = {
        'learning_rate': trial.suggest_categorical('learning_rate',[0.001,0.01,0.0001,0.02]),
        'epochs': trial.suggest_categorical('epochs', [100]),
        'batch_size': trial.suggest_categorical("batch_size", [128]),
        "cosine_factor" : trial.suggest_categorical("cosine_factor",[0.01,0.1,0.5,1.0]),
        "drop_out":trial.suggest_categorical("drop_out",[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]),
        "last_kernel_num":trial.suggest_int("last_kernel_num",1,20,step=1),
    }
    parser = argparse.ArgumentParser(description='Automatic parameter setting')
    parser.add_argument('--config_path', type=str, default=os.path.join(os.path.dirname(os.getcwd()), 'code/dl_param.json'))
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=p_args)

    tr = Train(args)
    train_args, rmse = tr.main(params_train)
    del tr
    global rmse_lst
    rmse_lst.append(rmse)
    print ("min_rmse:",np.min(rmse_lst))    
    print ("max_rmse:",np.max(rmse_lst))
    print ("mean_rmse:",np.mean(rmse_lst))
    return rmse



def main(args):
    if args.optuna:
        logger = logging.getLogger()
        logger.info("Train with the OPTUNA framework")
        study = optuna.create_study(direction="minimize", study_name=args.study_name,
                                    load_if_exists=True, sampler=optuna.samplers.TPESampler(),
                                    storage='sqlite:///db.sqlite1')
                
        study.optimize(train_optuna, n_trials=50)
        best_trial = study.best_trial
        logger.info("best_trial", best_trial)

        for key, value in best_trial.params.items():
            logger.info("%s: %s" % (key, value))
        global rmse_lst
        print ("rmse_lst",rmse_lst)
    else:
        Train(args).main()



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='Automatic parameter setting')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(os.getcwd()), 'code/dl_param.json'))
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)

    print("args", args)
    global rmse_lst
    rmse_lst=[]
    main(args) 



