import argparse
import os
import pprint
from multiprocessing import Process, Queue
from collections import OrderedDict
from turtle import pos

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
from tqdm import tqdm
import numpy as np
import time

import _init_paths
import models
from config import get_cfg, update_config
from core.evaluator import Evaluator
from dataset import make_test_dataloader
from utils.logging import create_checkpoint, setup_logger
from utils.transforms import get_multi_scale_size, resize_align_multi_scale, get_final_preds
from utils.nms import oks_nms



def parse_args():
	parser = argparse.ArgumentParser(description='Test CID')
	# general
	parser.add_argument('--cfg',
						help='experiment configure file name',
						required=True,
						type=str)
	parser.add_argument('opts',
						help="Modify config options using the command-line",
						default=None,
						nargs=argparse.REMAINDER)
	parser.add_argument('--gpus',
						help='gpu ids for eval',
						default='0',
						type=str)
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	cfg = get_cfg()
	update_config(cfg, args)

	final_output_dir = create_checkpoint(cfg, 'valid')
	logger, _ = setup_logger(final_output_dir, 0, 'valid')
	logger.info(pprint.pformat(args))
	logger.info(cfg)

	# cudnn related setting
	cudnn.benchmark = cfg.CUDNN.BENCHMARK
	torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
	torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

	_,test_dataset=make_test_dataloader(cfg)

	total_size=len(test_dataset)
	model=models.create(cfg.MODEL.NAME,cfg,is_train=False)

	if cfg.TEST.MODEL_FILE:
		logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
		model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
	else:
		model_state_file = os.path.join(final_output_dir, "model_best.pth.tar")
		logger.info("=> loading model from {}".format(model_state_file))
		model.load_state_dict(torch.load(model_state_file))

	gpu_list=args.gpus.split(',')
	num_gpus=len(gpu_list)
	os.environ['CUDA_VISIBLE_DEVICES']=gpu_list[0]

	assert num_gpus==1

	model=model.cuda()
	model.eval()

	durations=[]
	transforms= torchvision.transforms.Compose(
		[
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(
				mean=[0.485,0.456,0.406],
				std=[0.229,0.224,0.225]
			)
		]
	)

	with torch.no_grad():
		all_preds=[]
		print('start')
		for idx ,batch_inputs in enumerate(test_dataset):
			img_id = batch_inputs['image_id'].item()
			image=batch_inputs['image'][0].cpu().numpy()
			base_size, center, scale = get_multi_scale_size(
				image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
			)
			image_resized,center,scale=resize_align_multi_scale(image,cfg.DATASET.INPUT_SIZE,1.0,1.0)
			image_resized=transforms(image_resized)
			
			inputs=[{'image':image_resized}]
			torch.cuda.synchronize()
			start_time=time.perf_counter()
			instances=model(inputs)

			if 'poses' in instances:
				poses=instances['poses'].cpu().numpy()
				scores=instances['scores'].cpu().numpy()
				poses = get_final_preds(poses, center, scale, [base_size[0], base_size[1]])
				# perform nms
				keep, _ = oks_nms(poses, scores, cfg.TEST.OKS_SCORE, np.array(cfg.TEST.OKS_SIGMAS) / 10.0)

				for _keep in keep:
					all_preds.append({
						"keypoints": poses[_keep][:, :3].reshape(-1, ).astype(float).tolist(),
						"image_id": img_id,
						"score": float(scores[_keep]),
						"category_id": 1
					})
				
			torch.cuda.synchronize()
			end=time.perf_counter()-start_time
			durations.append(end)
			if (idx+1)%1000==0:
				print("process: [{}/{}] fps: {:.3f}".format(idx,len(test_dataset),1/np.mean(durations[500:])))
			
	latency=np.mean(durations[500:])
	fps=1/latency
	print("speed: {:.4f}s FPS: {:.2f}".format(latency,fps))



if __name__ == "__main__":
	main()