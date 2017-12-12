import torch.utils.data as data
import torchvision.transforms as transforms

# from model.model_L1lossOnly import GRNN_trainer
from model.model_toy_conv import GRNN_trainer
from src.dataloader_toy import Polygons
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data option
    parser.add_argument('--train_root', default='/home/jisu/Desktop/GRNN/data/toy')
    parser.add_argument('--test_root', default='/home/jisu/Desktop/GRNN/data/toy')
    parser.add_argument('--train_subjectlist', type=list, default=[1])  # [1, 5, 6, 7, 8, 9])
    parser.add_argument('--test_subjectlist', type=list, default=[5])
    parser.add_argument('--actionlist', type=list, default=['Walking'])
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--outf', default='./results')
    parser.add_argument('--heatpointsize', type=int, default=9)
    parser.add_argument('--seqlen', type=int, default=3)
    parser.add_argument('--framerate', type=int, default=6)
    # parser.add_argument('--num_joint', type=int, default=10)

    # Model option
    parser.add_argument('--npf', type=int, default=32)
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lb', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.5)

    # Save option
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--savedir', default='./results')

    # GPU option
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    opt = parser.parse_args()

    return opt

opt = parse_opt()
print opt

train_dataset = Polygons(opt.train_root)
train_dataloader = data.DataLoader(train_dataset, batch_size=opt.batchsize, num_workers=int(opt.workers), shuffle=True)
test_dataset = Polygons(opt.test_root)
test_dataloader = data.DataLoader(test_dataset, batch_size=opt.batchsize, num_workers=int(opt.workers), shuffle=True)

GRNN_trainer(opt, train_dataloader, test_dataloader)