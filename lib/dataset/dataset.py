import logging
import os
import os.path
from random import choice
import torch
import cv2
import numpy as np
import random

logger = logging.getLogger(__name__)

class PoseDataset(torch.utils.data.Dataset):
	def __init__(self, cfg, is_train, transform=None, target_generator=None):
		super(PoseDataset,self).__init__()
		self.root = cfg.DATASET.ROOT
		self.dataset = cfg.DATASET.DATASET
		self.file=cfg.DATASET.FILE_NAME
		from pycocotools.coco import COCO
		self.image_set = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
		self.is_train = is_train
		self.transform = transform
		self.input_size=cfg.DATASET.INPUT_SIZE
		self.output_size=cfg.DATASET.OUTPUT_SIZE
		self.coco = COCO(os.path.join(self.root, 'annotations', '{}_{}.json'.format(self.file, self.image_set)))
		self.ids = list(self.coco.imgs.keys())

		if is_train:
			if self.dataset=='coco':
				self.filter_for_annotations()

		self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
		self.output_size = cfg.DATASET.OUTPUT_SIZE
		self.heatmap_generator = target_generator

	def filter_for_annotations(self,min_kp_anns=1):
		print('filter for annotations (min kp=%d) ...', min_kp_anns)

		#filter for image with iscrowd!=1 and num_keypoints>0

		def filter_image(image_id):
			ann_ids=self.coco.getAnnIds(imgIds=image_id)
			anns=self.coco.loadAnns(ann_ids)

			anns=[ann for ann in anns if not ann.get('iscrowd')]
			if not anns:
				return False
			kp_anns=[ann for ann in anns if 'keypoints' in ann and any(v>0.0 for v in ann['keypoints'][2::3]) and ann['area']>32**2]
			return len(kp_anns)>=min_kp_anns
		self.ids=[image_id for image_id in self.ids if filter_image(image_id)]

	def __len__(self):
		return len(self.ids)

	def _get_image_path(self, file_name):
		if self.dataset=='coco':
			images_dir=self.root
		else:
			images_dir = os.path.join(self.root, 'images')
		if self.dataset == 'coco': images_dir = os.path.join(images_dir, '{}'.format(self.image_set))
		return os.path.join(images_dir, file_name)

	def __getitem__(self,index):
		coco=self.coco
		img_id=self.ids[index]
		file_name=coco.loadImgs(img_id)[0]['file_name']
		img=cv2.imread(self._get_image_path(file_name),cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
		img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		results={}
		if self.is_train:
			ann_ids=coco.getAnnIds(imgIds=img_id)
			target=coco.loadAnns(ann_ids)
			anno=[obj for obj in target]
			img_info=self.coco.loadImgs(img_id)[0]
			
			anno=[obj for obj in anno if obj['iscrowd']==0 and obj['num_keypoints']>0]

			num_people=len(anno)
			bboxs=np.zeros((num_people,4,2))
			keypoints=np.zeros((num_people,self.num_keypoints,3))
			
			
			for i,obj in enumerate(anno):
				keypoints[i, :, :3] = np.array(obj['keypoints']).reshape([-1, 3])
				bboxs[i, :, 0], bboxs[i, :, 1] = obj['bbox'][0], obj['bbox'][1]
				bboxs[i, 1, 0] += obj['bbox'][2]
				bboxs[i, 2, 1] += obj['bbox'][3]
				bboxs[i, 3, 0] += obj['bbox'][2]; bboxs[i, 3, 1] += obj['bbox'][3]


			if self.transform:
				img,ori_image,keypoints,ori_kp,bboxs=self.transform(img,keypoints,bboxs)
			

			inst_heatmaps=self.get_inst_annos(keypoints,bboxs)

			ratio=self.input_size/self.output_size
			offset=(ori_kp[:,:,:2]-(keypoints[:,:,:2].astype(int)*ratio))/ratio #:,:,2

			#print("offset",offset,'ori_kp',ori_kp,'keypoint',keypoints,sep='\n')
			inst_heatmaps = np.concatenate(inst_heatmaps, axis=0)
			results['instance_heatmap']=torch.from_numpy(inst_heatmaps)
			results['image']=img 
			results['img_id']=img_id
			results['ori_image']=torch.from_numpy(ori_image)
			results['offset']=torch.from_numpy(offset)
		else:
			results['image']=torch.from_numpy(img)
			results['image_id']=img_id
		return results

	def get_inst_annos(self, keypoints, bbox):
		inst_heatmaps= []
		for i in range(keypoints.shape[0]):
			inst_heatmap= self.heatmap_generator(keypoints[i:i+1, :, :], bbox[i:i+1, :, :])
			inst_heatmaps.append(inst_heatmap[None, :, :, :])
		return  inst_heatmaps


	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}'.format(self.root)
		return fmt_str