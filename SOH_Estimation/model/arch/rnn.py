import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN_Text(nn.Module):
    def __init__(self, kernel_sizes, kernel_num, class_num, embed_dim, last_kernel_num, drop_out, task_type, **params):
        super(RNN_Text, self).__init__()

        self.embed_dim= embed_dim  # 
        self.class_num = class_num # 1

        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.last_kernel_num = last_kernel_num
        self.drop_out_num = drop_out
        self.task_type = task_type

        self.convs1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels = self.embed_dim, out_channels = self.kernel_num, kernel_size = K, padding = int((K-1)/2)),
            nn.ReLU(),
            nn.Conv1d(in_channels = self.kernel_num, out_channels = self.last_kernel_num, kernel_size = K, padding = int((K-1)/2)))
            for K in self.kernel_sizes])
        self.convs2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels = self.embed_dim, out_channels = self.kernel_num, kernel_size = K, padding = int((K-1)/2)),
            nn.ReLU(),
            nn.Conv1d(in_channels = self.kernel_num, out_channels = self.last_kernel_num, kernel_size = K, padding = int((K-1)/2)))
            for K in self.kernel_sizes])

        self.dropout = nn.Dropout(self.drop_out_num)
        self.fc_aspect = nn.Linear(self.embed_dim, self.last_kernel_num)
        self.fc_cla = nn.Linear(len(self.kernel_sizes)*self.last_kernel_num, self.class_num)
        self.fc_reg = nn.Linear(len(self.kernel_sizes)*self.last_kernel_num, 1)


        embedding_size = 3
        hidden_size = 128
        num_layers = 3

        self.t_rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc_rnn = nn.Linear(hidden_size * embed_dim, 1)

    def forward(self, feature):


        batch_size = feature.size(0)
        feature = feature.to(torch.float32)
        outputs, hiddens = self.t_rnn(feature)
        outputs = outputs.reshape(batch_size, -1)

        return self.fc_rnn(outputs)
