import torch.utils.data as data
import torchvision.transforms as transforms

# from model.model_L1lossOnly import GRNN_trainer
from model.model_vector import GRNN_trainer
from src.Config import parse_opt
from src.dataloader_posemodified import human36

opt = parse_opt()
print opt

train_dataset = human36(opt, train=True, transform=transforms.ToTensor())
train_dataloader = data.DataLoader(train_dataset, batch_size=opt.batchsize, num_workers=int(opt.workers), shuffle=True)
test_dataset = human36(opt, train=False, transform=transforms.ToTensor())
test_dataloader = data.DataLoader(test_dataset, batch_size=opt.batchsize, num_workers=int(opt.workers), shuffle=True)

GRNN_trainer(opt, train_dataloader, test_dataloader)