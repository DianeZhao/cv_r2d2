import pdb
from queue import Full
import torch.nn as nn
import torch.nn.functional as F
from  nets.sampler import FullSampler
import random
from mlosses import distance_matrix_vector
from mlosses import loss_HardNet
#####warp,and get ground truth match
def RandomSubsample(feat1,feat2to1,crop_pixel=39):

    """
    feat1:  (B, C, H, W)   pixel-wise features extracted from img1
    feat2:  (B, C, H, W)   pixel-wise features extracted from img2
    """
    #random crop
    length=crop_pixel//2
    center_h=random.randint(length,192-length)
    center_w=random.randint(length,192-length)
    feat1_sub=feat1[:,:, center_w-length:center_w+length, center_h-length:center_h+length]
    feat2_sub=feat2to1[:,:, center_w-length:center_w+length, center_h-length:center_h+length]
    
    return  feat1_sub,feat2_sub
        
# def get_distance_matrix(feat1_sub,feat2_sub):
#     """
#     feat1:  (B, C, H, W)   pixel-wise features extracted from img1
#     feat2:  (B, C, H, W)   pixel-wise features extracted from img2
#     """
#     B,C,H, W = feat1_sub.shape
#     feat1 = feat1_sub.permute(0,2,3,1).clone().reshape(B*H*W,C)
#     feat2 = feat2_sub.permute(0,2,3,1).clone().reshape(B*H*W,C)
        


class HardnetLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, crop_pixel=39):
        nn.Module.__init__(self)
        self.crop_pixel=crop_pixel
        
    def forward(self, descriptors, aflow, **kw):
        # subsample things
        self.reliability=kw.get('reliability')
        self.sampler=FullSampler()#in order to use the warp function
        self.feat2to1, _ ,self. conf2to1=self.sampler._warp(descriptors,self.reliability,aflow)#def _warp(self, feats, confs, aflow)
        self.feat1,_=descriptors#to get feat of image1
        feat1_sub,feat2_sub=RandomSubsample(self.feat1,self.feat2to1,crop_pixel=39)
        B,C,H,W=feat1_sub.size()
        feat1_reshape = feat1_sub.permute(0,2,3,1).clone().reshape(B*H*W,C)
        feat2_reshape = feat2_sub.permute(0,2,3,1).clone().reshape(B*H*W,C)
        mtx=distance_matrix_vector(feat1_reshape,feat2_reshape)
        print("mtx",mtx.size())
        loss=loss_HardNet(feat1_reshape, feat2_reshape, anchor_swap = False, anchor_ave = False,\
            margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin")
        
     
        #print("loss",loss)
        return loss