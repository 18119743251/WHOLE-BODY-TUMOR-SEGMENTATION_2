import numpy
import torch
import torch.nn as nn
import h5py
import os
import json
import numpy as np
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from src.loss import DiceLoss
#from monai.metrics import DiceMetric, compute_meandice
from src.Unet3D.unet3d import UNet
import torch.optim as opt
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
#import cv2
from scipy.ndimage.interpolation import zoom
import torchvision.transforms as T
import random
#from torch.cuda.amp import GradScaler, autocast
#import einops

from monai.inferers import sliding_window_inference
from src.metrics import non_zero_acc, np_dice, dice_score
from src.loss import FocalLoss, TverskyLoss, FocalTverskyLoss, ComboLoss
from src.boundary_loss import SurfaceLoss
BCE = torch.nn.BCELoss()

class Trainer3d():
    def __init__(self,
                 model,train_set,test_set,opts, log_dir, model_path, 
                 device, shape, loss, pretrained=False, pre_path=None, 
                 loss_weight = [1], middle_boundary=False):
        self.model = model  # neural net
        # device agnostic code snippet
        self.device = device
        gpus = [0, 1]
        print(self.device)
        if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
          print(f"Let's use {torch.cuda.device_count()} GPUs!")
          self.model = nn.DataParallel(self.model, device_ids=gpus)  # 将模型对象转变为多GPU并行运算的模型
        self.model.to(gpus[0])
        print("1")# gyq
        self.epochs = opts['epochs']
        self.optimizer = torch.optim.AdamW(model.parameters(), opts['lr'], weight_decay=1e-5, amsgrad=True)
        print("2")# gyq
        self.criterion = loss                    # loss function
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=opts['batch_size'],
                                                        shuffle=True)
        print("3")# gyq
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=opts['batch_size'],
                                                       shuffle=False)
        print("start tb")# gyq
        self.tb = SummaryWriter(log_dir=log_dir)
        print("end tb")# gyq
        self.best_loss = 0
        self.maxdice = 0
        self.shape = shape
        self.model_path = './model_weights/modelUnetr_pp_tumor_128_RGB_MA.pth'
        self.loss_weight = loss_weight
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.middle_boundary = middle_boundary
        if self.middle_boundary:
            self.boundary = SurfaceLoss(**{'idc': 1})
        if pretrained:
            self.model = torch.load(pre_path, map_location=device)
            self.model.train()

    def joint_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool3d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def train(self):
        for epoch in range(self.epochs):
            print('the num of epoch: ',epoch)
            self.model.train() #put model in training mode
            self.tr_loss = []
            self.tr_dice = []
            print('It to begain train')
            #for j in range(3):
            self.output = []
            ii = 0
            for i, (pt, ct, labels) in tqdm(enumerate(self.train_loader),
                                                   total = len(self.train_loader)):
                pt, ct, labels = pt.to(self.device), ct.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                labels = torch.tensor(labels, dtype=torch.float32)
                pt = pt.cuda()
                ct = ct.cuda()
                labels = labels.cuda()
                #......................................
                x = pt.detach().cpu().numpy()
                y = labels.detach().cpu().numpy()
                yy = y.flatten()
                xx = x.flatten()
                for index, value in enumerate(xx):
                    if value != 0:
                        ii += 1
                        break
                #..................................
                outputs = self.model(pt,ct)
                #outputs2 = torch.tensor(outputs)
                #print('the shape of outputs is :', outputs2.size())
                #print('the type of is:', type(outputs))
                loss = self.joint_loss(outputs, labels)
                if len(self.criterion) > 1:
                    for i in range(1, len(self.criterion)):
                        cr = self.criterion[i]
                        loss += (self.loss_weight[i] * cr(outputs, labels)).item()
                # if self.middle_boundary and epoch > self.epochs/3:
                #     loss += (0.02 * self.boundary(outputs, labels)).item()
                loss.backward()
                self.optimizer.step()
                self.tr_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                x = predicted.detach().cpu().numpy()
                y = labels.detach().cpu().numpy()
                y = y[:, 0, :, :, :]
                dice_0, dice_1 = np_dice(x,y)
                self.tr_dice.append(dice_0)
                if (i+1) % 100 ==0:
                   print('The num of train: ',i+1)
            print('The train_loss : ',np.mean(self.tr_loss)) # 这里是求了个平均损失值，用np.mean()来求的
            print('the num 1 is:', ii)
            self.tb.add_scalar("Train Loss dice", np.mean(self.tr_loss), epoch)
            self.tb.add_scalar("Train dice", np.nanmean(self.tr_dice), epoch)
            #self.test(epoch) # run through the validation set,此处先不进行测试
            self.tb.close()
            self.output = outputs
       # print('开始存储模型')
            #torch.save((self.model).state_dict(), 'modelunetr_pp_to_threechannel.pth')
            self.test(self.output)
    # def dice(predict, target): # 自定义一个Dice
    #     if torch.is_tensor(predict):
    #         predict = torch.sigmoid(predict).data.cpu().numpy()
    #     if torch.is_tensor(target):
    #         target = target.data.cpu().numpy()
    #
    #     predict = numpy.atleast_1d(predict.astype(numpy.bool))  # 转一维数组
    #     target = numpy.atleast_1d(target.astype(numpy.bool))
    #
    #     intersection = numpy.count_nonzero(predict & target)  # 计算非零个数
    #
    #     size_i1 = numpy.count_nonzero(predict)
    #     size_i2 = numpy.count_nonzero(target)
    #
    #     try:
    #         dice = 2. * intersection / float(size_i1 + size_i2)
    #     except ZeroDivisionError:
    #         dice = 0.0
    #
    #     return dice
            
    def test(self, output):

            self.model.eval()    # puts model in eval mode - not necessary for this demo but good to know
            self.test_loss = []
            self.test_dice = []
            self.test_acc = []
            self.test_acc_all = []
            # self.model2 = model2
            xs = []
            ys = []
            c = 0
            total = 0
            #zero = torch.zeros_like(output)
            #one = torch.ones_like(output)
            print('It to begain test')
            for i, (pt,ct,labels) in enumerate(self.test_loader):
                
                pt, ct, labels = pt.to(self.device),ct.to(self.device),labels.to(self.device)
                #print('data size is', data.size())
                #print('labels size is', labels.size())
                with torch.no_grad():
                    labels = torch.tensor(labels, dtype=torch.float32)
                    #data = data.cuda()
                    pt = pt.cuda()
                    ct = ct.cuda()
                    labels = labels.cuda()
                    x = [pt,ct]
                    y_hat = sliding_window_inference(x, roi_size=(128,128,128), sw_batch_size=2,
                                                     predictor=self.model)

                    #outputs = self.model(pt,ct)
                    #  outputs = torch.nn.functional.softmax(outputs, 1) BCEW中已有softmax，两次使用导致网络不收敛
                loss = self.joint_loss(y_hat, labels)
                if len(self.criterion) > 1:
                    for i in range(1, len(self.criterion)):
                        cr = self.criterion[i]
                        loss += (self.loss_weight[i] * cr(y_hat, labels)).item()
                self.test_loss.append(loss.item())
                #self.sigmod = nn.Sigmoid()
                #output2 = self.sigmod(outputs)
                #print('outputs size is', outputs.size(), zero.size())
                zero = torch.zeros_like(y_hat)
                one = torch.ones_like(y_hat)
                
                output3 = torch.where(y_hat < 0.5, zero, y_hat)
                output4 = torch.where(output3 > 0.5, one, output3)

                x = output4.detach().cpu().numpy()
                y = labels.detach().cpu().numpy()

                self.test_acc.append((x == y).mean())
                #print(self.test_acc) #这里的test_acc是对0 labels，这个0标签的准确率，所以才那么高
                self.test_acc_all.append(non_zero_acc(x, y))
                #y = y[:, 0, :, :, :]
                xs.append(x)
                ys.append(y)
            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)
            dice_0 = dice_score(xs,ys)
            self.test_dice.append(dice_0)
            #print('the test_dice is:')
            #print(self.test_dice)
            mean_by_cases = []
            #print(total)
            for l in range(len(xs)):
                mean_by_cases.append(dice_score(xs[l], ys[l]))
            mean_by_cases = np.nanmean(mean_by_cases)
            print(' test loss: {},'
                                                'test dice non(1 ):{} ,'
                                                'test dice all(1 and 0 ):{} ,'
                                                'test dice for test_acc_all non:{} '
                                                .format(
                   np.mean(self.test_loss),
                                               np.mean(self.test_dice),
                                               np.nanmean(self.test_acc),
                                               np.nanmean(self.test_acc_all))
                 )
            self.maxdice = max(self.maxdice, np.nanmean(self.test_dice))
            print('The best dice is:----------',self.maxdice)
            #test dice non:Dice metric for label 1 in test data----self.test_dice
            #test dice all:0 and 1                             ----self.test_acc
            #test acc none: Test accuracy on label 1           ----self.test_acc_all
            # print('epoch: {}, test dice non: {}, test acc all: {}, test acc non {}, test dice mean by case {}'.format(
            #       epoch+1, np.nanmean(self.test_dice), np.mean(self.test_acc),np.nanmean(self.test_acc_all), mean_by_cases
            #      ))
            # self.tb.add_scalar("Val Loss", np.mean(self.test_loss), epoch)
            # self.tb.add_scalar("Val dice", np.nanmean(self.test_dice), epoch)
            # self.tb.add_scalar("Val acc", np.mean(self.test_acc), epoch)
            # self.tb.add_scalar("Val acc_all", np.mean(np.nanmean(self.test_acc_all)), epoch)
            # self.tb.add_scalar("Dice mean by case", mean_by_cases, epoch)
            if mean_by_cases > self.best_loss:
                self.best_loss = mean_by_cases
                torch.save(self.model, self.model_path)
