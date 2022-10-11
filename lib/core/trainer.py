import logging
import time
import torch

class Trainer(object):
	def __init__(self, cfg, model, rank, output_dir, writer_dict):
		self.model = model
		self.output_dir = output_dir
		self.rank = rank
		self.print_freq = cfg.PRINT_FREQ

	def train(self, epoch, data_loader, optimizer):
		logger = logging.getLogger("Training")

		batch_time = AverageMeter()
		data_time = AverageMeter()
		heatmap_loss_meter = AverageMeter()
		score_loss_meter = AverageMeter()
		offset_loss_meter = AverageMeter()

		self.model.train()


		print('epoch_lr: ',optimizer.state_dict()['param_groups'][0]['lr'])

		end = time.time()
		for i, batched_inputs in enumerate(data_loader):
			data_time.update(time.time() - end)

			loss_dict = self.model(batched_inputs)


			loss = 0
			num_images = len(batched_inputs)

			if 'heatmap_loss' in loss_dict:
				heatmap_loss = loss_dict['heatmap_loss']
				heatmap_loss_meter.update(heatmap_loss.item(), num_images)
				loss = loss+heatmap_loss
			
			if 'score_loss' in loss_dict:
				score_loss = loss_dict['score_loss']
				score_loss_meter.update(score_loss.item(), num_images)
				loss = loss+score_loss

			if 'offset_loss' in loss_dict:
				offset_loss = loss_dict['offset_loss']
				offset_loss_meter.update(offset_loss.item(), num_images)
				loss = loss+offset_loss


			optimizer.zero_grad()
			loss.backward()
			#for name, parms in self.model.named_parameters():
				#print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

			optimizer.step()

			batch_time.update(time.time() - end)
			end = time.time()

			if i % self.print_freq == 0 and self.rank == 0:
				msg = 'Epoch: [{0}][{1}/{2}]\t' \
					  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
					  'Speed: {speed:.1f} samples/s\t' \
					  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
					  '{heatmap_loss}{score}{offset_loss}\t' \
					  .format(
						epoch, i, len(data_loader),
						batch_time=batch_time,
						speed=num_images / batch_time.val,
						data_time=data_time,
						heatmap_loss=_get_loss_info(heatmap_loss_meter, 'heatmap_loss'),
						score=_get_loss_info(score_loss_meter, 'score_loss'),
						offset_loss=_get_loss_info(offset_loss_meter,'offset_loss')
						)
				logger.info(msg)

def _get_loss_info(meter, loss_name):
	msg = '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(name=loss_name, meter=meter)
	return msg

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count if self.count != 0 else 0
