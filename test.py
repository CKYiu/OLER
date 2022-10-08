import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
import os
from scipy import misc
from utils.data import test_dataset
from Network import OLER

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = OLER()
model = nn.DataParallel(model.cuda(), device_ids=[0])
model.load_state_dict(torch.load('./model/OLER-T.pth'))
model.cuda()
model.eval()

data_path = './dataset/'
valset = ['ECSSD', 'PASCAL-S', 'HKU-IS', 'DUT-OMRON', 'DUTS']

for dataset in valset:
    save_path = './Results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = data_path + dataset + '/img/'
    gt_root = data_path + dataset + '/gt/'
    test_loader = test_dataset(image_root, gt_root, testsize=352)

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.array(gt).astype('float')
            gt = gt / (gt.max() + 1e-8)
            image = Variable(image).cuda()

            r1, pre, e1, e2 = model(image)

            res = F.interpolate(pre, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            misc.imsave(save_path + name + '.png', res)
