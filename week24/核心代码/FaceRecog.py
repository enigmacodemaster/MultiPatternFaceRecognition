import torch
import os
import numpy as np
from config_mask import config
import os
# from validate_on_LFW import evaluate_lfw
from torch.nn.modules.distance import PairwiseDistance
import sys
# from Data_loader.Data_loader_facenet_mask import train_dataloader, test_dataloader, LFWestMask_dataloader
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pwd = os.path.abspath('./')


class Recognition:
	def __init__(self):
		from Models.Resnet34_attention import resnet34 as resnet34_cbam
		model_path = os.path.join(pwd, 'Model_training_checkpoints')
		self.model = resnet34_cbam(pretrained=True, showlayer= False, num_classes=128)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		x = [int(i.split('_')[4]) for i in os.listdir(model_path) if version in i]
		x.sort()
		for i in os.listdir(model_path):
			if (len(x)!=0) and ('epoch_'+str(x[-1]) in i) and (version in i):
				model_pathi = os.path.join(model_path, i)
				break
		model_pathi = os.path.join(model_path, 'model_34_triplet_mask.pt')
		print(model_path)
		
		if os.path.exists(model_pathi) and (version in model_pathi):
			if torch.cuda.is_available():
				model_state = torch.load(model_pathi)
			else:
				model_state = torch.load(model_pathi, map_location='cpu')
				start_epoch = model_state['epoch']
				print('loaded %s' % model_pathi)
		else:
			print('模型不存在！')
			sys.exit(0)
		
		if torch.cuda.is_available():
			self.model = self.model.cuda()
		
		self.l2_distance = PairwiseDistance(2).cuda()
		
		self.is_same = False
  		
    
	def detect(self, img1, img2, threshhold):
		img1 = img1.cuda()
		img2 = img2.cuda()
		output1, output2 = self.model(img1), self.model(img2)
		output1 = torch.div(output1, torch.norm(output1))
		output2 = torch.div(output2, torch.norm(output2))
		distance = self.l2_distance.forward(output1, output2)
		
		if distance < threshhold:
			self.is_same = True
		
		return self.is_same




