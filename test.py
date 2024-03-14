#!/usr/bin/python3
# coding=utf-8

import os
import sys

sys.path.insert(0, '/')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataset

from CEPNet_Model import CEPNet

import time



class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg = Dataset.Config(datapath=path, snapshot=r'D:\Paper\object detection paper\salient object detection\A论文代码\epnet2\epnet\epnet.pth.49', mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net = Network()
        self.net.load_state_dict(torch.load(self.cfg.snapshot))
        self.net.train(False)
        self.net.cuda()

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                s12, s34_up, s5_up, sig12, sig34up, sig5up ,edge1, edge2= self.net(image)
                out = s12

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1, 2, 0).cpu().numpy() * self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()

    def save(self):
        with torch.no_grad():
            time_sum = 0
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                time_start = time.time()
                s12,s34, s5, s12_sig, s34_sig, s5_sig= self.net(image)
                time_end = time.time()
                out = s12
                pred1 = F.interpolate(out, size=shape, mode='bilinear')
                pred = (pred1[0, 0]*255).cpu().numpy()
                head = 'eval\maps\epnet-49/' +self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))

            print('Running time {:.5f}'.format(time_sum / self.loader.size))
            print('FPS {:.5f}'.format(self.loader.size / time_sum))


if __name__ == '__main__':
     for path in ['D:\MyDataset\salient object detection/ECSSD', 'D:\MyDataset\salient object detection/DUT-OMRON',
                  'D:\MyDataset\salient object detection/HKU-IS', 'D:\MyDataset\salient object detection\DUTS/DUTS-TE',
                  'D:\MyDataset\salient object detection/PASCAL-S']:

        t = Test(dataset, CEPNet, path)
        t.save()
        # t.show()
