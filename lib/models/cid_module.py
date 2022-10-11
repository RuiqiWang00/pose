import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import os

def build_iia_module(cfg):
	return IIA(cfg)

def build_gfd_module(cfg):
	return GFD(cfg)


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
	convs = []
	for _ in range(num_convs):
		convs.append(
			nn.Conv2d(in_channels, out_channels, 3, padding=1))
		convs.append(nn.ReLU(True))
		in_channels = out_channels
	return nn.Sequential(*convs)



class IIA(nn.Module):
	def __init__(self,cfg):
		super().__init__()
		self.device = torch.device(cfg.MODEL.DEVICE)
		self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
		self.num_convs=cfg.MODEL.IIA.NUM_CONVS
		self.in_channels = cfg.MODEL.IIA.IN_CHANNELS+2
		self.kernel_dim=cfg.MODEL.KERNEL_DIM
		self.dim=cfg.MODEL.IIA.DIM
		self.num_persons=cfg.MODEL.NUM_PERSONS
		self.num_groups=cfg.MODEL.IIA.NUM_GROUPS
		self.prior_prob = cfg.MODEL.BIAS_PROB
		expand_dim=self.dim*self.num_groups

		self.inst_convs=_make_stack_3x3_convs(self.num_convs,self.in_channels,self.dim)
		self.iam_conv=nn.Conv2d(self.dim,self.num_persons*self.num_groups,3,padding=1)

		self.fc=nn.Linear(expand_dim,expand_dim)

		self.person_kernel=nn.Linear(expand_dim,self.kernel_dim*self.num_keypoints)
		self.offset=nn.Linear(expand_dim,self.num_keypoints*2)
		self.score=nn.Linear(expand_dim,1)
		self.prior_prob = 0.01
		self._init_weights()

	def _init_weights(self):
		for m in self.inst_convs.modules():
			if isinstance(m,nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
		for module in [self.iam_conv, self.score]:
			nn.init.constant_(module.bias, bias_value)
		nn.init.normal_(self.iam_conv.weight, std=0.01)
		nn.init.normal_(self.score.weight, std=0.01)

		nn.init.normal_(self.person_kernel.weight, std=0.01)
		nn.init.constant_(self.person_kernel.bias, 0.0)



	def forward(self,features,batch_inputs=None):
		features=self.inst_convs(features) #B,C,H,W

		iam=self.iam_conv(features)
		iam_prob=iam.sigmoid_() #B,(4*N),H,W

		B,N=iam_prob.shape[:2]
		C=features.size(1)

		iam_prob=iam_prob.view(B,N,-1) #B,(4*N),(H*W)
		inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
		normalizer = iam_prob.sum(-1).clamp(min=1e-6)
		inst_features = inst_features / normalizer[:, :, None] #B,(17*N),C

		inst_features = inst_features.reshape(B, 4, N // 4, -1).transpose(1, 2).reshape(B, N // 4, -1)

		inst_features = F.relu_(self.fc(inst_features)) #inst_feature:B,N,expand_dim
		pred_kernel=self.person_kernel(inst_features).view(B,self.num_persons,self.num_keypoints,-1)
		pred_score=self.score(inst_features) #######
		pred_offset=self.offset(inst_features).view(B,self.num_persons,self.num_keypoints,2)
		instances={}
		instances.update({'instance_param':pred_kernel})
		instances.update({'instance_score':pred_score})
		instances.update({'instance_offset':pred_offset})
		'''
		instances: {
			instance_param : B,N,D (kernels for every instance)
			instance_score: B,N,1 (score for whether is a person)
			instance_offset: B,N,17,2
		}
		'''
		return instances


class GFD(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.device = torch.device(cfg.MODEL.DEVICE)
		self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
		self.in_channels = cfg.MODEL.GFD.IN_CHANNELS+2
		self.kernel_dim=cfg.MODEL.KERNEL_DIM
		self.channels = cfg.MODEL.GFD.CHANNELS
		self.out_channels = self.num_keypoints
		self.prior_prob = cfg.MODEL.BIAS_PROB
		self.num_convs=cfg.MODEL.GFD.NUM_CONVS
		self.dim=cfg.MODEL.GFD.DIM

		self.mask_convs=_make_stack_3x3_convs(self.num_convs,self.in_channels,self.dim)
		self.conv_down = nn.Conv2d(self.dim, self.kernel_dim, 1, 1, 0)
		self.c_attn = ChannelAtten()
		self._init_weights()
	def _init_weights(self):
		for m in self.mask_convs.modules():
			if isinstance(m,nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)



	def forward(self, features, instances):
		'''
		instances: {
			instances_param : B,N,C,D (kernels for every instance)
			instancess_score: B,N,1 (score for whether is a person)
		}
		'''
		features=self.mask_convs(features)
		global_features = self.conv_down(features)
		B,C,H,W=global_features.size()
		instance_params = instances['instance_param']
		c_instance_feats = self.c_attn(global_features, instance_params)
		pred_instance_heatmaps=c_instance_feats

		del instances['instance_param']
		instances.update({'instance_heatmap':pred_instance_heatmaps})

		return instances




class ChannelAtten(nn.Module):
	def __init__(self):
		super(ChannelAtten, self).__init__()

	def forward(self, global_features, instance_params):
		B, D, H, W = global_features.size()
		global_features=global_features.view(B,D,(H*W))
		B,N,C,D=instance_params.shape
		instance_params=instance_params.view(B,(N*C),D)
		result=torch.bmm(instance_params,global_features)
		result=result.view(B,N,C,H,W)
		return result

'''
class SpatialAtten(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(SpatialAtten, self).__init__()
		self.atn = nn.Linear(in_channels, out_channels)
		conv_in = 1
		self.conv = nn.Conv2d(conv_in, 1, 5, 1, 2)

	def forward(self, global_features, instance_params):
		
		
		instances: {
			instances_param : B,N,D (kernels for every instance)
			instancess_score: B,N,1 (score for whether is a person)
		}
		global_features: B,C,H,W
		
		
		#Actually it is a 1×1 conv
		B, C, H, W = global_features.size()
		instance_params = self.atn(instance_params)
		feats=torch.bmm(instance_params,global_features.reshape(B,C,-1)).reshape(-1,1,H,W)
		#mask: B,N,1,H,W
		mask = self.conv(feats).sigmoid().reshape(B,-1,1,H,W)
		result=mask*(global_features.reshape(B,1,C,H,W))
		#result: B,N,C,H,W
		return result
'''