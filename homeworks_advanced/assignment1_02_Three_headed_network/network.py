from collections import OrderedDict
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_conv_pooling = nn.Sequential(OrderedDict([
            ('title_conv1', nn.Conv1d(
                in_channels=hid_size,
                out_channels=hid_size,
                kernel_size=2)),
            ('title_relu1', nn.ReLU()),
            ('title_adapt_avg_pool', nn.AdaptiveAvgPool1d(output_size=1))
        ]))
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_conv_pooling = nn.Sequential(OrderedDict([
            ('full_conv1', nn.Conv1d(
                in_channels=hid_size,
                out_channels=hid_size,
                kernel_size=2)),
            ('full_relu1', nn.ReLU()),
            ('full_adapt_avg_pool', nn.AdaptiveAvgPool1d(output_size=1))
        ]))
        
        self.category_out = nn.Linear(in_features=n_cat_features, out_features=hid_size)

        # Example for the final layers (after the concatenation)
        self.bn = nn.BatchNorm1d(concat_number_of_features)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)
        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_conv_pooling(title_beg) # <YOUR CODE HERE>
        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.full_conv_pooling(full_beg) # <YOUR CODE HERE>        
        
        category = self.category_out(input3) # <YOUR CODE HERE>        
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.inter_dense(self.bn(concatenated))
        out = self.final_dense(out)
        # <YOUR CODE HERE>
        
        return out