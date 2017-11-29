from PIL import Image
import numpy as np
import os
import cv2
os.environ["CDF_LIB"] = "/home/jisu/Downloads/cdf36_4-dist/src/lib"
from spacepy import pycdf
from scipy.io import loadmat
import scipy
import h5py
import argparse

def make_4digits(numstr):
    for _ in range(4 - len(numstr)):
        numstr = '0'+numstr
    return numstr

def save_frame(video_reader, save_path, frame_num):
    hasframe, img = video_reader.read()
    if hasframe:
        img = cv2.resize(img, (128, 128))
        cv2.imwrite(save_path+make_4digits(str(frame_num))+'.jpg', img)
        return True
    else:
        return False

def save_seg(seg_reader, save_path, frame_num):
    if len(seg_reader['Masks']) == frame_num:
        return False
    st = seg_reader['Masks'][frame_num][0]
    seg = np.transpose(seg_reader[st][:] * 255)
    im = Image.fromarray(seg)
    im = im.resize((128, 128))
    im.save(save_path+make_4digits(str(frame_num))+'.png')
    return True

def Image_loader(num, folder, video_reader, seg_reader, opt):
    video_savepath = opt.savedir + '/S%d/128x128Images/'%num + folder + '/'
    seg_savepath = opt.savedir + '/S%d/128x128seg/'%num + folder + '/'
    if not os.path.isdir(video_savepath):
        os.makedirs(video_savepath)
    if not os.path.isdir(seg_savepath):
        os.makedirs(seg_savepath)
    frame_num = 0
    while len(seg_reader['Masks']) > frame_num:
        save_frame(video_reader, video_savepath, frame_num)
        save_seg(seg_reader, seg_savepath, frame_num)
        frame_num += 1




def pose_loader(root, subject_list, action_list):
    subject_pose = {}
    for num in subject_list:
        action_pose = {}
        pose_path = root + '/S%d/MyPoseFeatures/D2_Positions'%num
        for folder in os.listdir(pose_path):
            if folder.split()[0] in action_list or folder.split('.')[0] in action_list:
                file_path = os.path.join(pose_path, folder)
                action_pose[folder[:-4]] = pycdf.CDF(file_path)[0][...][0]
        subject_pose['S%d'%num] = action_pose
    return subject_pose

def Gen_Data(opt):
    for num in opt.subject_list:
        video_root = opt.root + '/S%d/Videos'%num
        seg_root = opt.root + '/S%d/MySegmentsMat/ground_truth_bs'%num
        posepath = opt.root + '/S%d/MyPoseFeatures/D2_Positions'%num

        subject_save_dir = os.path.join(opt.savedir, 'S%d'%num)
        for folder in os.listdir(posepath):
            if opt.action_list:
                if folder.split()[0] not in opt.action_list and folder.split('.')[0] not in opt.action_list:
                    continue
            # print folder
            video_path = os.path.join(video_root, folder[:-4]+'.mp4')
            seg_path = os.path.join(seg_root, folder[:-4]+'.mat')
            with h5py.File(seg_path, 'r') as seg_loader:
                video_loader = cv2.VideoCapture(video_path)
                Image_loader(num, folder[:-4], video_loader, seg_loader, opt)
            print '%s complete' % folder[:-4]
        print 'Subject %d complete'%num
    print 'Generation complete'



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Data option
    parser.add_argument('--root', default='/media/jisu/UUI/human36m')
    parser.add_argument('--subject_list', type=list, default=[1])  # [1, 5, 6, 7, 8, 9])
    parser.add_argument('--action_list', type=list, default=['Walking'])

    parser.add_argument('--savedir', default='./test')

    opt = parser.parse_args()

    Gen_Data(opt)