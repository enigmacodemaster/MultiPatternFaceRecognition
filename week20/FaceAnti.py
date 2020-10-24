import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'
import sys
sys.path.append("../week19/week19code-CVPR19-Face-Anti-spoofing/")
sys.path.append("/media/ouyang/0933e5eb-356b-415d-926e-e1114043a3bc/home/ouyang/RoadtoAI/MultiPatternRecognition/week19/week19code-CVPR19-Face-Anti-spoofing/process")
from process.data_fusion import *
from process.augmentation import *
from metric import *
from collections import OrderedDict
from model import FaceBagNet_model_A
import cv2


class FaceAnti:
	def __init__(self):
		from FaceBagNet_model_A import Net
		self.net = Net(num_class=2, is_first_bn=True)
		model_path = "./global_min_acer_model.pth"
		if torch.cuda.is_availabel():
			state_dict = torch.load(model_pth, map_location='cuda')
			
		else:
			state_dict = torch.load(model_pth, map_location='cpu')
		new_state_dict = OrderedDict()
		
		for k, v in state_dict.items():
			name = k[7:]
			new_state_dict[name] = v
		self.net.load_state_dict(new_state_dict)
		
		if torch.cuda.is_availabel():
			self.net = self.net.cuda()
		
		
	def classify(self, color):
		return self.detect(color)
	
	def detect(self, color):
		color = cv2.resize(color, (RESIZE_SIZE, RESIZE_SIZE))
		
		def color_augmentor(image, target_shape=(64, 64, 3), is_infer=False):
			if is_infer:
				augment_img = iaa.Sequential([
					iaa.Fliplr(0),
				])
			
			image = augment_img.augment_image(image)
			image = TTA_36_cropps(image, target_shape)
			
			return image
		
		color = color_augmentor(color, target_shape=(64,64,3), is_infer=True)
		n = len(color)
		
		color = np.concatenate(color, axis=0)
		
		image = color
		image = np.tranpose(image, (0, 3, 1, 2)) # change dims
		image = image.astype(np.float32)
		image = image / 255.0
		input_image = torch.FloatTensor(image)
		
		if (len(input_image.size()) == 4) and torch.cuda.is_available():
			input_image = input_image.unsqueeze(0).cuda()
		elif (len(input_image.size()) == 4) and not torch.cuda.is_available():
			input_image = input_image.unsqueeze(0)
		
		b, n, c, w, h = input_image.size()
		input_image = input_image.view(b * n, c, w, h)
		
		if torch.cuda.is_availabel():
			input_image = input_image.cuda()
		
		with torch.no_grad():
			logit, _, _ = self.net(input_image)
			logit = logit.view(b, n, 2)
			logit = torch.mean(logit, dim = 1, keepdim = False)
			prob = F.softmax(logit, 1)
		
		print('probabilistic: ', prob)
		print('predict: ', np.argmax(prob.detach().cpu().numpy()))
		return np.argmax(prob.detach().cpu().numpy())
			
			
if __name__ == '__main__':
	FA = FaceAnti()
	image = cv2.imread('./humanFace.jpg', 1)
	FA.detect(img)
	
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
		
