import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os

class Resblock(nn.Module):
    def __init__(self, indepth, outdepth, stride=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(indepth, outdepth, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv3 = nn.Conv2d(outdepth, outdepth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outdepth)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class ModuleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ModuleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=1, bias=False)
        self.res = Resblock(out_channel, out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.res(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.module1 = ModuleBlock(3, opt.ngf)
        self.module2 = ModuleBlock(opt.ngf, opt.ngf * 2)
        self.module3 = ModuleBlock(opt.ngf * 2, opt.ngf * 4)
        self.module4 = ModuleBlock(opt.ngf * 4, opt.ngf * 8)
        self.module5 = ModuleBlock(opt.ngf * 8, opt.ngf * 16)
        self.module6 = ModuleBlock(opt.ngf * 16, opt.ngf * 32)
    def forward(self, x):
        x_1 = self.module1(x)
        x_1 = self.pool(x_1)  # 32x32
        x_2 = self.module2(x_1)
        x_2 = self.pool(x_2)  # 16x16
        x_3 = self.module3(x_2)
        x_3 = self.pool(x_3)  # 8x8
        x_4 = self.module4(x_3)
        x_4 = self.pool(x_4)  # 4x4
        x_5 = self.module5(x_4)
        x_5 = self.pool(x_5)  # 2x2
        x_ = self.module6(x_5)
        x_ = self.pool(x_)  # 2x2

        return x_, [x_1, x_2, x_3, x_4, x_5]

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(opt.ngf * 32, opt.ngf * 16, 2, 2)
        self.module7 = ModuleBlock(opt.ngf * 32, opt.ngf * 16)
        self.deconv2 = nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 8, 2, 2)
        self.module8 = ModuleBlock(opt.ngf * 16, opt.ngf * 8)

        self.deconv3 = nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 2, 2)
        self.module9 = ModuleBlock(opt.ngf * 8, opt.ngf * 4)
        self.deconv4 = nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 2, 2)
        self.module10 = ModuleBlock(opt.ngf * 4, opt.ngf * 2)
        self.deconv5 = nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 2, 2)
        self.module11 = ModuleBlock(opt.ngf * 2, opt.ngf)
        self.deconv6 = nn.ConvTranspose2d(opt.ngf, opt.ngf, 2, 2)
        self.module12 = ModuleBlock(opt.ngf, 3)
    def forward(self, x, x_pack):
        x_ = self.deconv1(x)
        x_ = torch.cat([x_, x_pack[4]], dim=1)
        x_ = self.module7(x_)
        x_ = self.deconv2(x_)
        x_ = torch.cat([x_, x_pack[3]], dim=1)
        x_ = self.module8(x_)
        x_ = self.deconv3(x_)
        x_ = torch.cat([x_, x_pack[2]], dim=1)
        x_ = self.module9(x_)

        x_ = self.deconv4(x_)
        x_ = torch.cat([x_, x_pack[1]], dim=1)
        x_ = self.module10(x_)
        x_ = self.deconv5(x_)
        x_ = torch.cat([x_, x_pack[0]], dim=1)
        x_ = self.module11(x_)

        x_ = self.deconv6(x_)
        # x_ = torch.cat([x_, x_3], dim=1)
        x_ = self.module12(x_)  # + x[:, :3, :, :]
        return x_

class GRNNcell(nn.Module):
    def __init__(self, opt):
        super(GRNNcell, self).__init__()
        self.encoder = Encoder(opt)
        self.fc_hh = nn.Linear(opt.ngf * 32, opt.ngf * 32)
        self.fc_xh = nn.Linear(opt.ngf * 32, opt.ngf * 32)
        self.decoder = Decoder(opt)
    def forward(self, x, hidden):
        [out, pack] = self.encoder(x)
        out = out.squeeze()
        # print self.fc_xh(out).size(), self.fc_hh(hidden).size()
        hidden = self.fc_xh(out) + self.fc_hh(hidden)
        hidden_ = hidden.unsqueeze(2).unsqueeze(3)
        out = self.decoder(hidden_, pack)
        return out, hidden

class GRNN(nn.Module):
    def __init__(self, opt):
        super(GRNN, self).__init__()
        self.opt = opt
        self.grnn_cell = GRNNcell(opt)
        self.train = True
        self.fc = nn.Linear(100, opt.ngf * 32)

    def Train(self):
        self.train = True
    def Eval(self):
        self.train = False

    def forward(self, x, hidden):
        out = []
        hidden = self.fc(hidden)
        if self.train == True:
            for x_ in x:
                [y, hidden] = self.grnn_cell(x_, hidden)
                out.append(y)
        else:
            for _ in range(self.opt.seqlen):
                [x, hidden] = self.grnn_cell(x, hidden)
                out.append(x)
        return out

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv = nn.Sequential(nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),     #64
#                                   nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),    #32
#                                   nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),   #16
#                                   nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),   #8
#                                   nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),  #4
#                                   nn.Conv2d(256, 1, 4, 1), nn.Sigmoid()
#                                   )
#
#     def forward(self, x):
#         out = self.conv(x)
#         return out.view(-1, 1).squeeze(1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 16, (1, 1, 1), 1), nn.ReLU(),                                                          # 32
            nn.Conv3d(16, 16, (4, 3, 3), 1, padding=(0, 1, 1)), nn.ReLU(), nn.MaxPool3d((1, 2, 2), (1, 2, 2)),  # 32
            nn.Conv3d(16, 32, (1, 3, 3), 1, padding=(0, 1, 1)), nn.ReLU(), nn.MaxPool3d((1, 2, 2), (1, 2, 2)),  # 16
            nn.Conv3d(32, 64, (1, 3, 3), 1, padding=(0, 1, 1)), nn.ReLU(), nn.MaxPool3d((1, 2, 2), (1, 2, 2)),  # 8
            nn.Conv3d(64, 128, (1, 3, 3), 1, padding=(0, 1, 1)), nn.ReLU(), nn.MaxPool3d((1, 2, 2), (1, 2, 2)), # 4
            nn.Conv3d(128, 256, (1, 4, 4), 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out.squeeze())
        return out

class ImgDiscriminator(nn.Module):
    def __init__(self):
        super(ImgDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4
            nn.Conv2d(256, 512, 4), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out.squeeze())
        return out



def GRNN_trainer(opt, train_dataloader, test_dataloader):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)

    net = GRNN(opt).cuda()
    net.apply(weights_init)
    Dnet = Discriminator().cuda()
    IDnet = ImgDiscriminator().cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(Dnet.parameters(), lr=opt.lr * 0.3, betas=(opt.beta1, 0.999))
    optimizerID = optim.Adam(IDnet.parameters(), lr=opt.lr * 0.3, betas=(opt.beta1, 0.999))
    Dcriterion = nn.BCELoss().cuda()
    fake = 0
    real = 1

    for epoch in range(opt.niter):
        net.Train()
        for i, image in enumerate(train_dataloader):
            #######################
            # Train Discriminator #
            #######################
            # Real_sequence
            Dnet.zero_grad()
            IDnet.zero_grad()
            input_real = [seq.unsqueeze(2) for seq in image]
            input_real = Variable(torch.cat(input_real, dim=2).cuda())
            label = Variable(torch.FloatTensor(input_real.size(0)).fill_(real)).cuda()
            out = Dnet(input_real).squeeze()
            errD_real_vid = Dcriterion(out, label)
            errD_real_vid.backward(retain_graph=True)

            # Real_image
            input_real = [torch.cat([pre_img, post_img], dim=1) for pre_img, post_img in zip(image[:-1], image[1:])]
            input_real = Variable(torch.cat(input_real, dim=0)).cuda()
            out = IDnet(input_real)
            label = Variable(torch.FloatTensor(out.size(0)).fill_(real)).cuda()
            errD_real_img = Dcriterion(out, label)
            errD_real_img.backward(retain_graph=True)

            D_x = out.data.mean()

            # Fake_sequence
            input_g = [Variable(item.cuda()) for item in image]
            init_hidden = torch.FloatTensor(image[0].size(0), 100)
            init_hidden = Variable(init_hidden.random_(0, 1)).cuda()
            out = net(input_g[:-1], init_hidden)

            input = [input_g[0].unsqueeze(2)]
            for item in out:
                input.append(item.unsqueeze(2))
            input__ = torch.cat(input, dim=2)
            out_ = Dnet(input__).squeeze()
            # print out_
            D_g_1 = out_.data.mean()
            label = Variable(torch.FloatTensor(out_.size(0)).fill_(fake)).cuda()
            errD_fake_vid = Dcriterion(out_, label)
            errD_fake_vid.backward(retain_graph=True)

            # Fake_image
            input_ = [torch.cat([pre_img, post_img], dim=1) for pre_img, post_img in zip(input_g[:-1], out)]
            input_ = torch.cat(input_, dim=0)
            label = Variable(torch.FloatTensor(input_.size(0)).fill_(fake)).cuda()
            out_ = IDnet(input_)
            errD_fake_img = Dcriterion(out_, label)
            errD_fake_img.backward(retain_graph=True)

            errD = errD_real_vid + errD_real_img + errD_fake_vid + errD_fake_img
            optimizerD.step()
            optimizerID.step()

            ###################
            # Train Generator #
            ###################
            # Fake_sequence
            net.zero_grad()
            out_ = Dnet(input__).squeeze()
            D_g_2 = out_.data.mean()
            label = Variable(torch.FloatTensor(out_.size(0)).fill_(real)).cuda()
            errG_vid = Dcriterion(out_, label)
            errG_vid.backward(retain_graph=True)

            # Fake_image
            out_ = IDnet(input_)
            label = Variable(torch.FloatTensor(out_.size(0)).fill_(real)).cuda()
            errG_img = Dcriterion(out_, label)
            errG_img.backward()

            errG = errG_vid + errG_img
            optimizer.step()



            if i % 20 == 0:
                print('[%d/%d][%d/%d] Loss_G: %.4f , Loss_D: %.4f , D(X): %.4f , D(G(x)): %.4f / %.4f'
                      % (epoch, opt.niter, i, len(train_dataloader),
                         errG.data[0], errD.data[0], D_x, D_g_1, D_g_2))
            if i % 125 == 0:
                save_img_gt = [seq[0, :, :, :].unsqueeze(0) for seq in input_g]
                save_img_gt = torch.cat(save_img_gt, dim=0)
                save_img_out = [seq.squeeze()[0, :, :, :].unsqueeze(0) for seq in input]
                save_img_out = torch.cat(save_img_out, dim=0)

                folder_path = '%s/toy_train_samples/%d' % (opt.outf, epoch)
                if not os.path.isdir(folder_path):
                    os.makedirs(folder_path)
                vutils.save_image(save_img_gt.data,
                                  '%s/toy_train_samples/%d/%d_real_samples.png' % (opt.outf, epoch, i))
                vutils.save_image(save_img_out.data,
                                  '%s/toy_train_samples/%d/%d_generated_samples.png' % (opt.outf, epoch, i))
        net.Eval()
        for i, image in enumerate(test_dataloader):
            input = [Variable(item.cuda()) for item in image]
            init_hidden = torch.FloatTensor(image[0].size(0), 100)
            init_hidden = Variable(init_hidden.random_(0, 1)).cuda()
            out = net(input[0], init_hidden)
            out_video = [input_g[0]]
            for item in out:
                out_video.append(item)
            print 'TEST [%d] video generated'%i
            for j in range(3):
                # save_img_gt = [seq[j, :, :, :].unsqueeze(0) for seq in input]
                # save_img_gt = torch.cat(save_img_gt, dim=0)
                save_img_out = [seq[j, :, :, :].unsqueeze(0) for seq in out_video]
                save_img_out = torch.cat(save_img_out, dim=0)

                folder_path = '%s/toy_test_samples/%d' % (opt.outf, epoch)
                if not os.path.isdir(folder_path):
                    os.makedirs(folder_path)
                # vutils.save_image(save_img_gt.data,
                #                   '%s/test_samples/%d/%d_real_samples(%d).png' % (opt.outf, epoch, i, j))
                vutils.save_image(save_img_out.data,
                                  '%s/toy_test_samples/%d/%d_generated_samples(%d).png' % (opt.outf, epoch, i, j))
            if i==10:
                break


















