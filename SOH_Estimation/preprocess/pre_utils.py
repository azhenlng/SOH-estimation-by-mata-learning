import os
import re
import numpy as np
import pandas as pd
import time
from scipy import interpolate as ip
from sklearn import preprocessing
from collections import OrderedDict


class Normalizer:
    def __init__(self, dfs=None, params=None):
        """
        normalizer
        :param dfs: list contains dataframe
        """
        res = []
        if dfs is not None:
            res.extend(dfs)
            self.max_norm = 0
            self.min_norm = 0
            self.std = 0
            self.mean = 0
            self.compute_min_max(res)
        elif params is not None:
            self.max_norm = np.array(params['max_norm'])
            self.min_norm = np.array(params['min_norm'])
            self.std = np.array(params['std'])
            self.mean = np.array(params['mean'])
        else:
            raise Exception("df list not specified")

    def compute_min_max(self, res):
        """
        compute min and max value
        """
        column_max_all = np.max(res, axis=1)
        column_min_all = np.min(res, axis=1)
        column_std_all = np.std(res, axis=1)
        column_mean_all = np.mean(res, axis=1)
        self.max_norm = np.max(column_max_all, axis=0) + 0.00001
        self.min_norm = np.min(column_min_all, axis=0) - 0.00001
        self.std = np.mean(column_std_all, axis=0)
        self.mean = np.mean(column_mean_all, axis=0)

    def std_norm_df(self, df):
        """
        compute std value
        """
        return (df - self.mean) / np.maximum(1e-4, self.std)

    def norm_func(self, df, norm_name):
        """
        select normalizer function
        :param df: dataframe m * n
        :param norm_name: 归一化子类的前缀名
        :return: 调用子类的归一化函数的结果
        """
        return eval(norm_name.capitalize() + 'Normalizer.norm')(self, df)#capitalize() 将字符串的第一个字母变成大写,其他字母变小写;eval() 函数用来执行一个字符串表达式，并返回表达式的值


class ChunengNormalizer(Normalizer):

    def norm(self, df):
        """
        :param df: dataframe m * n
        :return: normalized df
        """
        return (df - self.min_norm) / (self.max_norm - self.min_norm)


class PredictResult:

    def __init__(self, cluster_nums, sigvolt_nums):
        """
        metadata['cell_value_list']
        metadata['cell_value_mean']
        """
        self.error_rate = None
        self.cluster_nums = cluster_nums
        self.sigvolt_nums = sigvolt_nums
        metadata = OrderedDict()
        metadata['cluster'] = []
        metadata['cell_volt'] = []
        metadata['cell_value_list'] = []
        metadata['cell_value_mean'] = []
        metadata['cell_label_list'] = []
        metadata['cell_label_mean'] = []
        for i in range(self.cluster_nums):
            # metadata['cluster'].append(i + 1)
            metadata['cell_value_list'].append([])
            metadata['cell_value_mean'].append([])
            metadata['cell_label_list'].append([])
            metadata['cell_label_mean'].append([])
            for j in range(self.sigvolt_nums):
                metadata['cell_value_list'][i].append([])
                metadata['cell_value_mean'][i].append([])
                metadata['cell_label_list'][i].append([])
                metadata['cell_label_mean'][i].append([])
        self.metadata = metadata

    def save_result(self, output_metadata, restored_outputs):
        for i in range(len(output_metadata['cluster'])):
            if output_metadata['cluster'][i] not in self.metadata['cluster']:
                self.metadata['cluster'].append(output_metadata['cluster'][i])
            if output_metadata['cell_volt'][i] not in self.metadata['cell_volt']:
                self.metadata['cell_volt'].append(output_metadata['cell_volt'][i])

            cell_index = self.metadata['cell_volt'].index(output_metadata['cell_volt'][i])
            cluster_index = self.metadata['cluster'].index(output_metadata['cluster'][i])
            self.metadata['cell_value_list'][cluster_index][cell_index].append(restored_outputs[i][0])
            self.metadata['cell_label_list'][cluster_index][cell_index].append(output_metadata['label'].cpu().detach()[i][0])

    def calculate(self):
        for i in range(self.cluster_nums):
            for j in range(self.sigvolt_nums):
                self.metadata['cell_value_mean'][i][j] = np.mean(self.metadata['cell_value_list'][i][j])
                self.metadata['cell_label_mean'][i][j] = np.mean(self.metadata['cell_label_list'][i][j])

        residual = np.array(self.metadata['cell_value_mean']) - np.array(self.metadata['cell_label_mean'])
        error_rate_list = abs(residual)/np.array(self.metadata['cell_label_mean'])
        self.error_rate = np.mean(error_rate_list)

    def get_rank(self):
        capacity_result = []
        for i in range(self.cluster_nums):
            for j in range(self.sigvolt_nums):
                cell_result = [int(self.metadata['cluster'][i]),int(re.findall("\d+", self.metadata['cell_volt'][j])[0]),
                            self.metadata['cell_value_mean'][i][j], self.metadata['cell_label_mean'][i][j]]
                capacity_result.append(cell_result)

        return pd.DataFrame(capacity_result,
                            columns=['cluster', 'cell_index', 'predict_capacity', 'label']).sort_values(
            by='predict_capacity', ascending=False)

class EDPredictResult:

    def __init__(self, cluster_nums, sigvolt_nums):
        """
        metadata['cell_value_list']
        metadata['cell_value_mean']
        """
        self.error_rate = None
        self.cluster_nums = cluster_nums
        self.sigvolt_nums = sigvolt_nums
        metadata = OrderedDict()
        metadata['cluster'] = []
        metadata['exp_num'] = []
        metadata['cell_value_list'] = []
        metadata['cell_value_mean'] = []
        metadata['cell_label_list'] = []
        metadata['cell_label_mean'] = []
        for i in range(self.cluster_nums):
            # metadata['cluster'].append(i + 1)
            metadata['cell_value_list'].append([])
            metadata['cell_value_mean'].append([])
            metadata['cell_label_list'].append([])
            metadata['cell_label_mean'].append([])
            for j in range(self.sigvolt_nums):
                metadata['cell_value_list'][i].append([])
                metadata['cell_value_mean'][i].append([])
                metadata['cell_label_list'][i].append([])
                metadata['cell_label_mean'][i].append([])
        self.metadata = metadata

    def save_result(self, output_metadata, restored_outputs):
        for i in range(len(output_metadata['cluster'])):
            if output_metadata['cluster'][i] not in self.metadata['cluster']:
                self.metadata['cluster'].append(output_metadata['cluster'][i])
            if output_metadata['exp_num'][i] not in self.metadata['exp_num']:
                self.metadata['exp_num'].append(output_metadata['exp_num'][i])

            cell_index = self.metadata['exp_num'].index(output_metadata['exp_num'][i])
            cluster_index = self.metadata['cluster'].index(output_metadata['cluster'][i])
            self.metadata['cell_value_list'][cluster_index][cell_index].append(restored_outputs[i][0])
            self.metadata['cell_label_list'][cluster_index][cell_index].append(output_metadata['label'].cpu().detach()[i][0])

    def calculate(self):
        for i in range(self.cluster_nums):
            for j in range(self.sigvolt_nums):
                self.metadata['cell_value_mean'][i][j] = np.mean(self.metadata['cell_value_list'][i][j])
                self.metadata['cell_label_mean'][i][j] = np.mean(self.metadata['cell_label_list'][i][j])

        residual = np.array(self.metadata['cell_value_mean']) - np.array(self.metadata['cell_label_mean'])
        error_rate_list = abs(residual)/np.array(self.metadata['cell_label_mean'])
        self.error_rate = np.mean(error_rate_list)

    def get_rank(self):
        capacity_result = []
        for i in range(self.cluster_nums):
            for j in range(self.sigvolt_nums):
                cell_result = [int(self.metadata['cluster'][i]),int(re.findall("\d+", self.metadata['exp_num'][j])[0]),
                            self.metadata['cell_value_mean'][i][j], self.metadata['cell_label_mean'][i][j]]
                capacity_result.append(cell_result)

        return pd.DataFrame(capacity_result,
                            columns=['cluster', 'cell_index', 'predict_capacity', 'label']).sort_values(
            by='predict_capacity', ascending=False)




class LabelNormalizer():
    """
    label normalizer
    """

    def __init__(self, label_path):
        """
        initailizing
        :param label_path: label path
        """
        label_lst = sorted(os.listdir(label_path))
        if os.path.join(label_path, label_lst[0]).endswith('.csv'):
            label_df = pd.read_csv(os.path.join(label_path, label_lst[0]), index_col='file_name')
        elif os.path.join(label_path, label_lst[0]).endswith('.feather'):
            label_df = pd.read_feather(os.path.join(label_path, label_lst[0]), )
        self.label_df = label_df

    def label_minmaxscaler(self, min_num=None, max_num=None):
        """
        :param min_num: min num
        :param max_num: max num
        :return: norm func
        """
        cell_list = [list(self.label_df)[i] for i in range(len(list(self.label_df))) if 'relative_capacity_' in list(self.label_df)[i]]
        label_df2 = self.label_df[[i for i in cell_list]]
        if not min_num:
            min_num = label_df2.min().min()
        if not max_num:
            max_num = label_df2.max().max()
        temp_label = [[max_num], [min_num]]
        label_normalizer = preprocessing.MinMaxScaler()
        label_normalizer.fit(temp_label)

        return label_normalizer

   

    def label_standardscaler(self):
        """
        :return: label norm func
        """
        cell_colum = [i for i in list(self.label_df) if 'relative_capacity' in i]
        cell_capacity = np.array(self.label_df[cell_colum])
        cell_capacity = cell_capacity.reshape(-1, 1)
        label_normalizer = preprocessing.StandardScaler()
        label_normalizer.fit(cell_capacity)

        return label_normalizer




class EDNormalizer():
    """
    label normlizer
    """

    def __init__(self):
        """
        :param label_path
        """
       

    def exp_minmaxscaler(self, min_num=None, max_num=None):
        """
        :param min_num
        :param max_num
        :return
        """
        temp_label = [[max_num], [min_num]]
        label_normalizer = preprocessing.MinMaxScaler()
        label_normalizer.fit(temp_label)

        return label_normalizer
