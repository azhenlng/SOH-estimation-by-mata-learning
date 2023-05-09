import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, kernel_sizes, kernel_num, class_num, embed_dim, last_kernel_num, drop_out, task_type, **params):
        super(CNN_Gate_Aspect_Text, self).__init__()

        self.embed_dim= embed_dim  # 
        self.class_num = class_num # 1

        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.last_kernel_num = last_kernel_num
        self.drop_out_num = drop_out
        self.task_type = task_type

        self.convs1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.last_kernel_num, kernel_size=K,
                      padding=int((K - 1) / 2))
            for K in self.kernel_sizes])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.last_kernel_num, kernel_size=K,
                      padding=int((K - 1) / 2))
            for K in self.kernel_sizes])

        self.dropout = nn.Dropout(self.drop_out_num)
        self.fc_aspect = nn.Linear(self.embed_dim, self.last_kernel_num)
        self.fc_reg = nn.Linear(len(self.kernel_sizes)*self.last_kernel_num, 1)

    def forward(self, feature):

        feature = feature.to(torch.float32)
        aspect_v = feature.transpose(1, 2).sum(1) / feature.transpose(1, 2).size(1)

        x = [torch.tanh(conv(feature)) for conv in self.convs1]
        
        y = [F.relu(conv(feature) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]

        x = [i*j for i, j in zip(x, y)]


        x0 = [F.max_pool1d(i, int(i.size(2))).squeeze(2) for i in x]

    
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)

        logit = self.fc_reg(x0)  
        
        return logit

