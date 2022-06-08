#!/usr/bin/env python
from audioop import cross
import re
import torch
from mlosses import distance_matrix_vector, loss_HardNet
from nets.my_reliabilityloss import RandomSubsample
from nets.sampler import FullSampler


if __name__ == '__main__':
    descriptors=torch.rand(2,8,128,192,192)#dimension test, warped[2,8, 128, 192, 192]
    feat1_d,feat2_d=descriptors
    reliability=torch.rand(2,8,1,192,192)#dimension test, warped
    #flow=torch.rand(8,2,192,192)
    #sampler=FullSampler()
    #subsample=RandomSubsample(feat1_d,feat2_d,crop_pixel=39)
    feat1_sub,feat2_sub=RandomSubsample(feat1=feat1_d,feat2to1=feat2_d,crop_pixel=39)
    B,C,H,W=feat1_sub.size()
    feat1_reshape = feat1_sub.permute(0,2,3,1).clone().reshape(B*H*W,C)
    feat2_reshape = feat2_sub.permute(0,2,3,1).clone().reshape(B*H*W,C)
    mtx=distance_matrix_vector(feat1_reshape,feat2_reshape)
    print("mtx",mtx.size())
    loss=loss_HardNet(feat1_reshape, feat2_reshape, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin")
    
    
    
    
    # print("f1sub",feat1_sub.size())
    # print("f2sub",feat2_sub.size())#f1sub torch.Size([8, 128, 38, 38])


    #feat2to1, _ , conf2to1=sampler._warp(feats=descriptors,confs=reliability,aflow=flow)
    
    # print(torch.rand(4,4))
    # vector_anchor=torch.tensor([[0.8, 0.6, 0.5, 0.1],
    #     [0.30, 0.6, 0.3, 0.8],
    #     [0.2, 0.7, 0.5, 0.1],
    #     [0.1, 0.7, 0.8, 0.2]])
    # vector_positive=torch.tensor([[0.85, 0.8, 0.5, 0.2],
    #     [0.3, 0.8, 0.4, 0.7],
    #     [0.3, 0.7, 0.4, 0.2],
    #     [0.1, 0.8, 0.9, 0.3]])
    # distance_matrix_vector(vector_anchor, vector_positive)
    # loss_HardNet(vector_anchor, vector_positive, anchor_swap = True,  margin = 1.0, loss_type = "triplet_margin")
