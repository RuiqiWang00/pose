import torch
from torch import nn
import torch.nn.functional as F

from .backbone import build_backbone
from .cid_module import build_iia_module, build_gfd_module
from .loss import Loss


class CID(nn.Module):
	def __init__(self, cfg, is_train):
		super().__init__()
		self.device = torch.device(cfg.MODEL.DEVICE)
		self.backbone = build_backbone(cfg, is_train)
		self.iia = build_iia_module(cfg)
		self.gfd = build_gfd_module(cfg)

		self.score_loss_weight = cfg.LOSS.SCORE_LOSS_WEIGHT
		self.heatmap_loss_weight = cfg.LOSS.SINGLE_HEATMAP_LOSS_WEIGHT
		self.heatmap_cost_weight = cfg.LOSS.SINGLE_HEATMAP_COST_WEIGHT
		self.offset_loss_weight=cfg.LOSS.OFFSET_WEIGHT
		self.loss=Loss(hms_cost_weight=self.heatmap_cost_weight,hms_weight=self.heatmap_loss_weight,score_weight=self.score_loss_weight,offset_weight=self.offset_loss_weight)

		self.threshold=cfg.TEST.THRESHOLD
		self.kernel_size=cfg.TEST.KERNEL_SIZE

	def forward(self, batch_inputs):
		images = [x['image'].unsqueeze(0).to(self.device) for x in batch_inputs]
		images = torch.cat(images, dim=0)
		feats = self.backbone(images)
		coord=self.compute_coordinates(feats)
		feats=torch.cat([coord,feats],dim=1)
		instances = self.iia(feats,images)

		if self.training:
			instances = self.gfd(feats, instances)
			#instances:{
				#instance_heatmap: B,N,17,H,W
				#instance_score:B,N,1
			#}
			# bipartite matching
			heatmap_loss,score_loss,offset_loss=self.loss(instances,batch_inputs)
			heatmap_loss=heatmap_loss*self.heatmap_loss_weight
			score_loss=score_loss*self.score_loss_weight
			offset_loss=offset_loss*self.offset_loss_weight
			losses={}
			losses.update({'heatmap_loss':heatmap_loss})
			losses.update({'score_loss':score_loss})
			losses.update({'offset_loss':offset_loss})
			return losses
		else:
			results={}
			
			scores=instances['instance_score'].squeeze().sigmoid() #N
			param=instances['instance_param'] #1,N,D
			offset=instances['instance_offset'].squeeze(dim=0) #N,17,2

			target=(scores>self.threshold) 
			if target.shape[0]==0:
				return results

			scores=scores[target]
			param=param[:,target,:]
			offset=offset[target,:,:]

			instances['instance_param']=param

			instances = self.gfd(feats, instances)

			instance_heatmap=instances['instance_heatmap'].squeeze(dim=0).sigmoid()
			offset=offset.sigmoid() #N,17,2

			center_pool = F.avg_pool2d(instance_heatmap, 3, 1, 1)
			instance_heatmap=(instance_heatmap+center_pool)/2

			N,C,H,W=instance_heatmap.shape
			nms_instance_heatmap=instance_heatmap.flatten(2)

			vals,inds=torch.max(nms_instance_heatmap,dim=2)
			#val,ind: N,C
			x,y=inds%W,(inds/W).long()
			
			x, y = self.adjust(x, y, offset)
			#x,y:N,C
			vals=vals*scores.unsqueeze(dim=1)
			#score=score.unsqueeze(dim=0)
			#vals=score.view(N,1).expand_as(x)
			poses=torch.stack((x,y,vals),dim=2)

			scores=torch.mean(poses[:,:,2],dim=1)

			results.update({'poses':poses})
			results.update({'scores':scores})
			return results


	def adjust(self, res_x, res_y, offset):
		'''
		res_x,res_y: N,C
		offset: N,C,2
		'''
		res_x=res_x*4+offset[:,:,0]*4
		res_y=res_y*4+offset[:,:,1]*4
		return res_x, res_y
	def adjust_wo_offset(self, res_x, res_y, heatmaps):
		n, k, h, w = heatmaps.size()
		x_l, x_r = (res_x - 1).clamp(min=0), (res_x + 1).clamp(max=w-1)
		y_t, y_b = (res_y + 1).clamp(max=h-1), (res_y - 1).clamp(min=0)
		n_inds = torch.arange(n)[:, None].to(self.device)
		k_inds = torch.arange(k)[None].to(self.device)

		px = torch.sign(heatmaps[n_inds, k_inds, res_y, x_r] - heatmaps[n_inds, k_inds, res_y, x_l])*0.25
		py = torch.sign(heatmaps[n_inds, k_inds, y_t, res_x] - heatmaps[n_inds, k_inds, y_b, res_x])*0.25
		res_x, res_y = res_x.float(), res_y.float()
		x_l, x_r = x_l.float(), x_r.float()
		y_b, y_t = y_b.float(), y_t.float()
		px = px*torch.sign(res_x-x_l)*torch.sign(x_r-res_x)
		py = py*torch.sign(res_y-y_b)*torch.sign(y_t-res_y)

		res_x = res_x.float() + px
		res_y = res_y.float() + py

		return res_x, res_y

	@torch.no_grad()
	def compute_coordinates(self, x):
		h, w = x.size(2), x.size(3)
		y_loc = torch.linspace(-1, 1, h, device=x.device)
		x_loc = torch.linspace(-1, 1, w, device=x.device)
		y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
		y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
		x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
		locations = torch.cat([x_loc, y_loc], 1)
		return locations.to(x)
		