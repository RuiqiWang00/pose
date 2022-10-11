import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class Matcher(nn.Module):
	def __init__(self,
				hms_weight=2.0,
				score_weight=1.0,
				offset_weight=1.0) -> None:
		super().__init__()
		self.hms_weight=hms_weight
		self.score_weight=score_weight
		self.offset_weight=offset_weight

	def forward(self,
				pred_hms,
				gt_heatmaps,
				pred_scores,
				pred_offsets,
				gt_offsets):
		'''
		pred_hms:Tensor B,N,C,H,W
		gt_heatmaps:a list of Tensor ,the list length=B ,Tensor shape=n,C,H,W
		pred_scores:Tensor B,N,1
		pred_offsets: B,N,C,2
		gt_offsets:a list of Tensor ,the list length=B ,Tensor shape=n,C,2
		'''
		with torch.no_grad():
			B,N,C,H,W=pred_hms.shape
			pred_heatmaps=pred_hms.flatten(3)
			gt_heatmaps=[gt_hms.flatten(2) for gt_hms in gt_heatmaps]
			cost=list(map(self.single_cost,pred_heatmaps,pred_scores,gt_heatmaps,pred_offsets,gt_offsets))
			indices=self.match(cost)
			return indices

	def match(self,cost):
		device=cost[0].device
		cost=[c.permute(1,0).cpu().numpy() for c in cost]

		indices=[linear_sum_assignment(x)[1] for x in cost]
		indices=[torch.as_tensor(x,dtype=torch.long) for x in indices]
		return indices

	def single_cost(self,
					pred_hms,
					pred_scores,
					gt_hms,
					pred_offsets,
					gt_offsets):
		'''
		gt_hms: Tensor n,C,(H*W)
		'''
		heatmap_cost=self.sigmoid_hms_cost(pred_hms,gt_hms)
		N,n=heatmap_cost.shape
		gt_scores=torch.ones((n,1),dtype=pred_scores.dtype,device=pred_scores.device)
		score_cost=self.sigmoid_score_cost(pred_scores,gt_scores)
		mask=(gt_hms.sum(-1)==0) #n,C
		offsets_cost=self.sigmoid_offset_cost(pred_offsets,gt_offsets,mask)
		c = heatmap_cost*self.hms_weight+score_cost*self.score_weight+offsets_cost*self.offset_weight
		return c

	def sigmoid_offset_cost(self,
							pred_offsets,
							gt_offsets,
							mask):
		'''
		argues:
		pred_offsets:  N,C,2
		gt_offsets : n,C,2
		mask: n,C heatmap that has a keypoint==False,else True
		implement of L2 loss
		'''
		pred_offsets=pred_offsets.sigmoid().unsqueeze(dim=1)#N,1,C,2
		#gt_offsets n,C,2
		cost=torch.pow((pred_offsets-gt_offsets),2).sum(-1)#N,n,C
		cost[:,mask]=0.0
		num_kp=((~mask).float().sum(-1)).clamp(min=1) #n
		cost=cost.sum(-1)/(num_kp.unsqueeze(dim=0))/2.0
		return cost


	def sigmoid_score_cost(self,
						  pred_scores,
						  gt_scores,
						  alpha=0.25,
						  gamma=2.0):
		'''
		pred_scores:Tensor ,shape= N,1 without sigmoid
		gt_scores :Tensor ,shape=n,1
		return cost matrix shaped N,n
		simpler implement of focal loss, because the gt_scores is always  1.0
		'''
		N,n=pred_scores.shape[0],gt_scores.shape[0]
		p=torch.sigmoid(pred_scores)
		inputs=pred_scores.expand(N,n) #N,n
		targets=gt_scores.permute(1,0).expand(N,n) #N,n
		ce_loss=F.binary_cross_entropy_with_logits(
			inputs,targets,reduction="none"
		)
		loss=ce_loss*((1-p)**gamma)
		if alpha >0:
			loss=alpha*loss

		return loss


	def sigmoid_hms_cost(self,pred_hms,
				gt_heatmaps,
				beta=4.0,
				gamma=2.0):
		'''
		pred_hms: Tensor N,C,(H*W) without sigmoid
		gt_heatmaps: Tensor n,C,(H*W)
		return heatmap cost matrix shape=N,n
		implement of focal loss
		'''
		N,n=pred_hms.shape[0],gt_heatmaps.shape[0]
		mask_no_kp=(gt_heatmaps.sum(2)==0) #n,C
		num_kp=(~mask_no_kp).float().sum(-1).clamp(min=1) #n
		p=torch.sigmoid(pred_hms).unsqueeze(dim=1) #N,1,C,-1
		inputs=pred_hms.unsqueeze(dim=1).expand(-1,n,-1,-1)#N,n,C,-1
		targets=gt_heatmaps.unsqueeze(dim=0).expand(N,-1,-1,-1)#N,n,C,-1

		ce_loss=F.binary_cross_entropy_with_logits(
			inputs,targets,reduction="none"
		) #N,n,C,-1
		ce_loss[:,mask_no_kp,:]=0.0
		mask_eq1=gt_heatmaps.eq(1).float() #n,C,-1
		mask_neq1=1-mask_eq1
		p_t=((1-p)*mask_eq1+p*mask_neq1)**gamma #N,n,C,-1
		ratio=mask_eq1+(((1-gt_heatmaps)*mask_neq1)**beta) #n,C,-1
		cost=ce_loss*p_t*ratio #N,n,C,-1

		cost=(cost.sum(-1).sum(-1))/(num_kp.unsqueeze(0))
		return cost

class Loss(nn.Module):
	def __init__(self,
				matcher=Matcher,
				hms_cost_weight=2.0,
				hms_weight=2.0,
				score_weight=1.0,
				offset_weight=1.0) -> None:
		super().__init__()
		self.matcher=matcher(hms_weight=hms_cost_weight,
							score_weight=score_weight,
							offset_weight=offset_weight)
		self.hms_weight=hms_weight
		self.score_weight=score_weight
		self.score_loss=sigmoid_focal_loss
		self.offset_weight=offset_weight

	def forward(self,outputs,targets):
		pred_hms=outputs['instance_heatmap']
		pred_scores=outputs['instance_score']
		pred_offsets=outputs['instance_offset']
		gt_heatmaps=[v['instance_heatmap'] for v in targets]
		gt_offsets=[v['offset'] for v in targets]
		'''
		pred_hms :Tensor (pred_heatmaps) shape=B,N,C,H,W without sigmoid
		pred_scores: Tensor shape=B,N,1 without sigmoid
		pred_offsets:Tensor shape=B,N,C,2 without sigmoid
		gt_heatmaps:a list of Tensor ,length of list=B, Tensor shape=n,C,H,W
		gt_offsets: a list of Tensor, lenght =B,Tensor shape=n,C,2
		'''
		gt_heatmaps_and_masks_and_offsets=[self.remove_zeros(val,offset) for val,offset in zip(gt_heatmaps,gt_offsets)]
		gt_heatmaps_and_masks_and_offsets=list(map(list,zip(*gt_heatmaps_and_masks_and_offsets)))
		gt_heatmaps,masks,gt_offsets=gt_heatmaps_and_masks_and_offsets[0],gt_heatmaps_and_masks_and_offsets[1],gt_heatmaps_and_masks_and_offsets[2]
		#remember the indices is the pred_heatmap indice 
		indices=self.matcher(pred_hms,gt_heatmaps,pred_scores,pred_offsets,gt_offsets)

		hms=[pred_hms[i] for i in range(pred_hms.size(0))]
		gt_heatmaps=torch.cat(gt_heatmaps,dim=0)
		pred_hms=[hm[indice] for hm,indice in zip(hms,indices)]
		pred_hms=torch.cat(pred_hms,dim=0) #N_^,C,H,W
		N,C,H,W=gt_heatmaps.shape
		gt_heatmaps=gt_heatmaps.flatten(1) 
		mask=(gt_heatmaps.sum(-1)!=0.0) #N
		gt_heatmaps=gt_heatmaps[mask,:].view(-1,C,H,W)
		pred_hms=(pred_hms.flatten(1)[mask,:]).view(-1,C,H,W)

		pred_offsets=[pred_offsets[i] for i in range(pred_offsets.shape[0])]
		gt_offsets=torch.cat(gt_offsets,dim=0) #N,C,2
		pred_offsets=[offset[indice] for offset,indice in zip(pred_offsets,indices)]
		pred_offsets=torch.cat(pred_offsets,dim=0) #N,C,2
		pred_offsets,gt_offsets=pred_offsets[mask,:,:],gt_offsets[mask,:,:]
		offset_mask=(gt_heatmaps.flatten(2).sum(-1)==0.0) #N,C




		offset_loss=self.offset_loss(
			pred_offsets,gt_offsets,offset_mask
		)


		heatmap_loss=self.heatmap_loss(pred_hms,gt_heatmaps)

		pred_scores=[pred_scores[i] for i in range(pred_scores.size(0))]
		gt_scores=[torch.zeros_like(pred_score,requires_grad=False) for pred_score in pred_scores]
		indices=[indice[mask] for indice,mask in zip(indices,masks)]
		def f(score,indice):
			score[indice]=1.0
			return score
		gt_scores=[f(score,indice) for score,indice in zip(gt_scores,indices)]
		pred_scores=torch.cat(pred_scores,dim=0) #(B*N),1
		gt_scores=torch.cat(gt_scores,dim=0) #(B*N),1
		score_loss=self.score_loss(pred_scores,
								   gt_scores,
								   alpha=0.25,
								   gamma=2.0,
								   reduction="mean")


		return heatmap_loss,score_loss,offset_loss
		
	def offset_loss(self,
					pred_offsets,
					gt_offsets,
					offset_mask,
					):

		'''
		pred_offsets: Tensor N,C,2
		gt_offsets shape the same as pred_offsets
		offset_mask: N,C
		mask: n,C heatmap that has a keypoint==False,else True
		'''
		pred_offsets=pred_offsets.sigmoid()
		#print('pred_offsets',pred_offsets,'gt_offsets',gt_offsets)		
		num_kp=(~offset_mask).float().sum()
		if is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_kp)
		num_kp=torch.clamp(num_kp/get_world_size(),min=1).item()
		loss=torch.pow((pred_offsets-gt_offsets),2).sum(-1)
		loss[offset_mask]=0.0
		loss=loss.sum()/num_kp/2.0
		return loss


	def heatmap_loss(self,
					pred_hms,
					gt_heatmaps,
					beta=4.0,
					gamma=2.0):
		'''
		pred_hms:Tensor shape=N,C,H,W without sigmoid
		gt_heatmaps :Tensor shape the same as pred_hms
		'''
		N,C,H,W=pred_hms.shape
		pred_hms,gt_heatmaps=pred_hms.flatten(2),gt_heatmaps.flatten(2) #N,C,(H*W)
		p=torch.sigmoid(pred_hms)
		mask_nkp=(gt_heatmaps.sum(-1)==0) #N,C
		num_kp=(~mask_nkp).float().sum()
		if is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_kp)
		num_kp=torch.clamp(num_kp/get_world_size(),min=1).item()

		ce_loss=F.binary_cross_entropy_with_logits(
			pred_hms,gt_heatmaps,reduction='none'
		)#N,C,-1
		ce_loss[mask_nkp,:]=0.0
		mask_eq1=gt_heatmaps.eq(1).float() #N,C,-1
		mask_neq1=1-mask_eq1

		p_t=((1-p)*mask_eq1+p*mask_neq1)**gamma
		ratio=mask_eq1+(((1-gt_heatmaps)*mask_neq1)**beta)
		loss=(ce_loss*p_t*ratio).sum()/num_kp
		return loss

	def remove_zeros(self,hm,offset):
		'''
		hm: Tensor N,C,H,W 
		offset: Tensor N,C,2
		remove the person which has not a keypoint at all
		'''
		N,C,H,W=hm.shape
		mask=(torch.sum(hm.flatten(1),dim=1)!=0) #N
		gt_heatmaps=hm[mask]
		gt_offset=offset[mask]
		if gt_heatmaps.shape[0]==0:
			gt_heatmaps=torch.zeros((1,C,H,W),device=gt_heatmaps.device)
			gt_offset=torch.zeros((1,17,2),device=gt_heatmaps.device)
		N=gt_heatmaps.shape[0]
		mask=(torch.sum(gt_heatmaps.view(N,-1),dim=1)!=0)
		return (gt_heatmaps,mask,gt_offset)
