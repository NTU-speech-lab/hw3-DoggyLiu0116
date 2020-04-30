import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

args = {
      'ckptpath': './best_224.pth',
      'dataset_dir': './food-11/'
}
args = argparse.Namespace(**args)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            # in_channels = 3 是因為一開始有 RGB 三原色的 channal
            # out_channels = 64 代表 kernel數有 64 個
            # kernel_size = 3 代表 3x3 的 kernel
        # torch.nn.BatchNorm2d 看你 out_channels 有多少，就在之後做normalize
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
#             nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
#             nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
#             nn.Dropout(0.3),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
#             nn.Dropout(0.3),
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
#             nn.Dropout(0.4),
        )
        # fc 就是 fully connected feedforward network
        # nn.leanear(input, output)
            # 一開始輸入的 512*4*4 等同 flatten 後數量
            # 最後剩下11個類別如同我們所 label 的類別數
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

model_best = Classifier().cuda()
checkpoint = torch.load(args.ckptpath)
model_best.load_state_dict(checkpoint)
# 基本上出現 <All keys matched successfully> 就是有載入成功，但最好還是做一下 inference 確認 test accuracy 沒有錯。

# 讀取檔案的函式
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 224, 224, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        # x是一個四維的資料
        # 第一維: 代表第幾個data
        # 第二維: 第幾列
        # 第三維: 第幾行
        # 第四維: RGB
        x[i, :, :] = cv2.resize(img,(224, 224))
        # 檔名裡，底線前的數字代表label，底線後的可能只是編號沒什麼差
        if label:
            y[i] = int(file.split("_")[0])
    if label:
          return x, y
    else:
        return x

#分別將 training set、validation set、testing set 用 readfile 函式讀進來
workspace_dir = sys.argv[1]
print("Reading data")
# test 不用label，所以false，看readfile函式可發現它也指return x沒有return x, y
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))


#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    # 在 from torch.utils.data import DataLoader, Dataset 中的 DataLoader, Dataset
    # 需要 __len__ 及 __getitem__ 兩個函式
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
        
batch_size = 60

test_set_ver2 = ImgDataset(test_x, transform=test_transform)
test_loader_ver2 = DataLoader(test_set_ver2, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader_ver2):
        test_pred_ver2 = model_best(data.cuda())
        test_label_ver2 = np.argmax(test_pred_ver2.cpu().data.numpy(), axis=1)
        for y in test_label_ver2:
            prediction.append(y)

#將結果寫入 csv 檔
with open(sys.argv[2], 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))


