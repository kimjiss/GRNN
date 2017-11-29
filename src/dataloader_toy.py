import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import numpy as np
import os

def video_loader(path):
	transform = transforms.ToTensor()
	img_pack = []
	for i in range(4):
		frame_num = str(int(path.split('/')[-1][:-4]) + i)
		path_ = path.split('/')[:-1]
		paths = ''
		for folder in path_:
			paths = os.path.join(paths, folder)
		path_ = os.path.join(paths, frame_num + '.bmp')
		with open('/'+path_, 'rb') as f:
			with Image.open(f) as img:
				img = img.convert('RGB')
				img = transform(img)
				img_pack.append(img)
	return img_pack



def make_imglist(root):
	videos = []
	for file in os.listdir(root):
		if int(file[:-4])%4 == 0:
			videos.append(os.path.join(root, file))
	return videos

class Polygons(data.Dataset):
	def __init__(self, root):
		self.video_list = make_imglist(root)

	def __getitem__(self, idx):
		video_path = self.video_list[idx]
		video = video_loader(video_path)
		return video

	def __len__(self):
		return len(self.video_list)
