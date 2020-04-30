# Import需要的套件

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import sys

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
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
# test 不用label，所以false，看readfile函式可發現它也指return x沒有return x, y
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

#training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomRotation(15), #隨機旋轉圖片
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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
batch_size = 32
train_set = ImgDataset(train_x, train_y, train_transform)
# val_set 要用 test_transform ，是因為這邊 val_set 是 testing data
val_set = ImgDataset(val_x, val_y, train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


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

model = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) # optimizer 使用 Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epoch = 700

for epoch in range(num_epoch):
    # time相關套件只是為了顯示過多少時間
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    # data 代表從 train_loader 裡頭得到的張量，i就是idx
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        # \ 只是代表換行而已
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

"把train跟val合起來"
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_better = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss，可能有接到softmax
optimizer = torch.optim.SGD(model_better.parameters(), lr=0.01, momentum=0.9)

num_epoch = 700

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_better.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_better(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_better.eval()
prediction = []
delete_test = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        # test_pred 用 .shape得到的[128,11] 代表 [batch size, answer]
        test_pred = model_better(data.cuda())
        # test_pred 機率大於 0.8 或是 0.9 再放進 training data 

        # 用 torch.max 會得到 value 與 index，要value就如下後面接上 [0]即可
        test_value = list(torch.max(test_pred,1)[0])
        batch_max_mean = sum(test_value)/len(test_value)-9
        
        for d in range(0,test_pred.shape[0]):
            if float(test_value[d]) < int(batch_max_mean):
                delete_test.append(d+i*batch_size)

        # test_label 為 test_pred 中最大數值的index
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)


        # 把 test_label內容加到prediction的list內
        for y in test_label:
            prediction.append(y)
        
# list 轉為 np.array
test_y = np.array(prediction)
test_add_x=np.delete(test_x,delete_test, axis=0)
test_add_y=np.delete(test_y,delete_test, axis=0)

"把train跟val跟自己定義後的test合起來"
train_val_test_x = np.concatenate((train_val_x, test_add_x), axis=0)
train_val_test_y = np.concatenate((train_val_y, test_add_y), axis=0)
train_val_test_set = ImgDataset(train_val_test_x, train_val_test_y, train_transform)
train_val_test_loader = DataLoader(train_val_test_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss，可能有接到softmax
optimizer = torch.optim.SGD(model_best.parameters(), lr=0.01, momentum=0.9)
num_epoch = 700

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_test_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_test_set.__len__(), train_loss/train_val_test_set.__len__()))

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
with open("predict_sgd_self_adjust_model_revise_vgg19.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

