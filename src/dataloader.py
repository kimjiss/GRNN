import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import numpy as np
import os

def make_4digits(numstr):
    for _ in range(4 - len(numstr)):
        numstr = '0'+numstr
    return numstr

def pil_loader(path, seq_len, framerate, transform):
    img_pack = []
    for i in range(seq_len):
        frame_num = str(int(path[-8:-4]) + i*framerate)
        frame_num = make_4digits(frame_num)
        path_ = path[:-8] + frame_num + '.jpg'
        with open(path_, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
                img = img.resize((128, 128))
                img = transform(img)
                img_pack.append(img)
    return img_pack

def seg_loader(path, seq_len, framerate, transform):
    seg_pack = []
    for i in range(seq_len):
        frame_num = str(int(path[-8:-4]) + i*framerate)
        frame_num = make_4digits(frame_num)
        path_ = path[:-8] + frame_num + '.png'
        with open(path_, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('L')
                img = img.resize((128, 128))
                img = transform(img)
                seg_pack.append(img)
    return seg_pack

def pose_loader(path, seq_len, framerate):
    pose_pack = []
    for i in range(seq_len):
        frame_num = str(int(path[-8:-4]) + i*framerate)
        frame_num = make_4digits(frame_num)
        path_ = path[:-8] + frame_num + '.txt'
        with open(path_, 'rb') as f:
            data = f.read().split(' ')
            data = [float(x)/2 for x in data[:-1]]
        pose_pack.append(data)
    return pose_pack

def make_imglist(root, subject_list, action_list, seq_len, framerate):
    images = []
    for num in subject_list:
        img_path = root + '/S%d/128x128Images'%num
        pose_path = root + '/S%d/128x128Poses'%num
        seg_path = root + '/S%d/128x128seg' % num

        for folder in os.listdir(seg_path):
            # action = folder.split()[0]
            if folder.split()[0] in action_list or folder.split('.')[0] in action_list:
            # if action in action_list:
                img_folder_path = os.path.join(img_path, folder)
                seg_folder_path = os.path.join(seg_path, folder)
                pose_folder_path = os.path.join(pose_path, folder)
                for file in os.listdir(img_folder_path):
                    file_dir = os.path.join(img_folder_path, file)
                    pose_dir = os.path.join(pose_folder_path, file[:-4]+'.txt')
                    seg_dir = os.path.join(seg_folder_path, file[:-4] + '.png')

                    file_end_dir = os.path.join(img_folder_path, str(int(file[:-4]) + seq_len*framerate) + '.jpg')
                    if os.path.isfile(pose_dir):
                        if os.path.isfile(file_end_dir):
                            pair = (file_dir, pose_dir, seg_dir)
                            images.append(pair)
                    else:
                        raise LookupError("Direction %s doesnt exist."%pose_dir)
    print "Image, Pose directory loaded."
    return images

class human36(data.Dataset):
    def __init__(self, opt, train, transform=None):
        self.opt = opt
        if train==True:
            img_list = make_imglist(opt.dataroot, opt.train_subjectlist, opt.actionlist, opt.seqlen, opt.framerate)
        else:
            img_list = make_imglist(opt.dataroot, opt.test_subjectlist, opt.actionlist, opt.seqlen, opt.framerate)
        self.imgs = img_list
        self.transform = transform

    def __getitem__(self, idx):
        img_path, pose_path, seg_path = self.imgs[idx]
        image = pil_loader(img_path, self.opt.seqlen, self.opt.framerate, self.transform)
        pose = pose_loader(pose_path, self.opt.seqlen, self.opt.framerate)
        seg = seg_loader(seg_path, self.opt.seqlen, self.opt.framerate, self.transform)
        return image, pose, seg

    def __len__(self):
        return len(self.imgs)