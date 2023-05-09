import os
import numpy as np
import pandas as pd
import multiprocessing
import random
from scipy import interpolate as ip
from collections import OrderedDict

import sys


class SlidingWindowBattery:
    """
    sliding window
    """

    def __init__(self, data_path, file_list, exp_data, window_len, interval, jobs, ram, sigvolt_nums, window_limit, interpolate, partition,partition_num, is_interpolate,mode = 'train' ):
        """
        :param data_path:  str
        :param file_list:  list
        :param exp_data:  bool
        :param window_len:  int
        :param interval: int
        :param jobs: int
        :param ram: bool
        :param sigvolt_nums: int
        :param window_limit:  int
        :param partition:  bool
        :param partition_num:  int
        """

        self.data_path = data_path
        self.exp_data = exp_data
        self.battery_dataset = []
        # self.data_temp_lst = sorted(os.listdir(data_path))
        self.data_temp_lst = file_list
        if len(self.data_temp_lst) == 0:
            print('.csv or .feather file not found')
            exit()
        self.window_len = window_len
        self.interval = interval
        self.ram = ram
        self.sigvolt_nums = sigvolt_nums
        self.window_limit = window_limit
        self.interpolate = interpolate
        self.is_interpolate = is_interpolate
        self.sigvolt_nums = sigvolt_nums
        self.partition = partition  
        self.mode = mode
        if self.exp_data:
            if "NatEnergy" in self.data_path or "NEall" in self.data_path:
                self.column_filter = ['volt', 'current','temp' ,'quantity','dq_dv']
            else :
                self.column_filter = ['volt', 'current','temp' ,'quantity','dqdv']
            self.data_lst = [i for i in self.data_temp_lst]
        
        print("Loading dataset",self.data_path)
        print("Using parallel loader for %d files; this takes about 1min per 50,000 files." % len(self.data_lst))

        if self.partition :
            self.p = partition_num
            self.p_len= len(self.data_lst)//self.p
            # print("Data will be split into %s partition"%self.p)
            self.num=[]
            for file in self.data_lst:
                results=[]
                self.battery_dataset=[]
                results.append(SlidingWindowBattery.pool_map([self, file]))
                for lst in [i[0] for i in results]:
                    self.battery_dataset.extend(lst)
                self.num.append(len(self.battery_dataset))

            self.numlst=[]
            self.num_sum=0
            for i in range(len(self.data_lst)):
                self.num_sum+=self.num[i]
                if (i+1) % self.p_len ==0:
                    self.numlst.append(self.num_sum)
                    self.num_sum =0
            self.p_num=0
            self.p_datapath=self.data_lst[0:self.p_len]
            self.amount=[]
            for i in range(len(self.numlst)):
                self.amount.append(sum(self.numlst[:i]))
            self.amount.append(sum(self.numlst))
            results=[]
            self.battery_dataset=[]

        else:
            self.p_datapath=self.data_lst
            self.numlst=None

        try:
            if jobs == 1:
                results = [self.pool_map([self, file]) for file in self.p_datapath if self.check(file)] 
            else:
                pool = multiprocessing.Pool(jobs)
                results = pool.map(SlidingWindowBattery.pool_map, [[self, file] for file in self.p_datapath if self.check(file)])
                pool.close()
                pool.terminate()
                pool.join()

            for lst in [i[0] for i in results]:

                self.battery_dataset.extend(lst)

            print("data length %d" % len(self.battery_dataset))
        except RuntimeError as e:
            print(e)
            
     
    def check(self,file):
        try :
            df = pd.read_csv(os.path.join(self.data_path, file))
        except Exception as e:
            print(file)
        
        if "NCA" in self.data_path or "NCM" in self.data_path or "NCM_NCA" in self.data_path:
            data_len = 50
        else:
            data_len = 100
        df1 = df.iloc[:data_len, :]
        df_np_q = df1['quantity'].to_numpy()#
        if np.max(df_np_q) == np.inf or np.max(df_np_q) == 0:
            return False

        return True
            
    def check1(self,file):
        df = pd.read_csv(os.path.join(self.data_path, file))
        df_np_volt = df['Voltages'].to_numpy()
        if df_np_volt.size == 0:
            return False
        if np.max(df_np_volt) == np.inf or np.max(df_np_volt) == 0:
            return False
        return True
    
    def dqdv(self,df):
        a = [0]
        for i in range(len(df)-1):
            res = (df['quantity'][i+1] - df['quantity'][i])/(df['volt'][i+1] - df['volt'][i])
            a.append(res)
        return np.array(a)

    @staticmethod
    def pool_map(args):
        """
        file 
        """
        try:
            self, file = args[0], args[1]
            return_lst = []

            if os.path.join(self.data_path, file).endswith('.csv'):

                try :
                    df = pd.read_csv(os.path.join(self.data_path, file))
                except Exception as e:
                    print(file)

                df = df.fillna(0) 
                if "TsinghuaWX" in self.data_path or "Tshall" in self.data_path:
                    df['dqdv'] = np.array(self.dqdv(df))
            elif os.path.join(self.data_path, self.data_lst[0]).endswith('.feather'):
                df = pd.read_feather(os.path.join(self.data_path, file))

            if self.is_interpolate:
                df = sampling(df, self.data_path, self.interpolate)

            if self.exp_data:
                name = os.path.splitext(file)[0].split('_')
                metadata = OrderedDict()
                metadata['file'] = os.path.splitext(file)[0]
                metadata['exp_num'] = name[0]+"_"+name[1]
                metadata['charge_num'] = name[2]
                metadata['cluster'] = '0'

                if self.mode == 'train':

                    if "TsinghuaWX" in self.data_path or "Tshall" in self.data_path:
                        metadata['label'] = np.array([df['cap'][0]])
                    else:
                        metadata['label'] = np.array([df['capacity'][0]])
                else:
                    metadata['label'] = None
                cell_df = pd.DataFrame(df, columns=self.column_filter)
                cell_df = cell_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
                if cell_df.shape[0] >= self.window_len:
                    window_num = int((cell_df.values.shape[0] - self.window_len) / self.interval) + 1
                    index = 0
                    while index < window_num and index < self.window_limit:
                        start_num=random.randint(0,len(cell_df)-self.window_len)
                        return_lst.append((np.array(cell_df.iloc[start_num:start_num + self.window_len, :]),metadata))
                        index += 1

            
            return [return_lst]

        except Exception as e:
            print(e)
    def __len__(self):

        if self.partition :
            return sum(self.numlst)
        else:
            return len(self.battery_dataset)
    def getnumlst(self):
        return self.numlst
    def __getitem__(self, idx):
        if self.partition:
            if idx>=self.amount[self.p_num+1] or (idx<self.amount[1] and self.p_num==(self.p-1)):
                if idx<self.amount[1] and self.p_num==(self.p-1):
                    self.p_num=0
                else:
                    self.p_num+=1
                # print ("loading next partition %s"%self.p_num)
                self.p_datapath = self.data_lst[self.p_num*self.p_len:(self.p_num+1)*self.p_len]
                results=[]
                self.battery_dataset=[]
                self.df_lst = {}
                for file in self.p_datapath:
                    results.append(SlidingWindowBattery.pool_map([self, file]))
                for lst in [i[0] for i in results]:
                    self.battery_dataset.extend(lst)
            idx = idx-self.amount[self.p_num]
            if idx >= len(self.battery_dataset):
                idx = random.randint(0,len(self.battery_dataset))

        sig_data, label = self.battery_dataset[idx]
        return sig_data, label



class PreprocessNormalizer:
    """
    data normalizer
    """

    def __init__(self, dataset, norm_name=None, normalizer_fn=None):
        """
        :param dataset: SlidingWindowBattery
        :param norm_name: chunengNormalizer
        :param normalizer_fn: norm func
        """
        self.dataset = dataset
        self.norm_name = norm_name
        self.normalizer_fn = normalizer_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        df, label = self.dataset[idx][0], self.dataset[idx][1]
        if self.normalizer_fn is not None:
            df = self.normalizer_fn(df, self.norm_name)
        return df, label

    def get_column(self):
        df = self.dataset[1][0]
        return list(df.columns)


def sampling(df, data_path, interval=1, columns_name='timestamp',):
    """
    :param df: dataframe n * m
    :param interval
    :param columns_name: interval value
    :return:
    """

    df[columns_name] -= df[columns_name].min()
    if "NEall" in data_path:
        df[columns_name] = df[columns_name]*60
    target_time_idx = np.arange(df[columns_name].min() + 1,
                                df[columns_name].max() - 1,
                                interval)

    if len(target_time_idx) <= 10 or df.isnull().values.any():
        new_df = pd.DataFrame(df[:1])
        new_df.columns = list(df.columns)
        return new_df

    df = df.drop_duplicates(subset=[columns_name])
    original_time = df[columns_name].values
    data_array = df.values
    f = ip.interp1d(original_time, data_array, axis=0)
    interpolated = f(target_time_idx)
    new_df = pd.DataFrame(interpolated)
    new_df.columns = list(df.columns)

    return new_df

