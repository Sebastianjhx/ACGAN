import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import pickle
import copy
import tqdm
import matplotlib.gridspec as gridspec
import os


def save_model(model, filename):  # 保存为CPU中可以打开的模型
    state = model.state_dict()
    x = state.copy()
    for key in x:
        x[key] = x[key].clone().cpu()
    torch.save(x, filename)
# state_dict作为python的字典对象将每一层的参数映射成tensor张量

def showimg(images, count):
    images = images.to('cpu')
    images = images.detach().numpy()
    images = images[[5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95]]
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    grid_length = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(4, 4))
    width = images.shape[2]
    gs = gridspec.GridSpec(grid_length, grid_length, wspace=0, hspace=0)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
    #     plt.tight_layout()
    # plt.savefig(r'./CGAN/images/%d.png' % count, bbox_inches='tight')


def loadmnist(batch_size):  # MNIST图片的大小是28*28
    trans_img = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST('./data', train=True, transform=trans_img, download=True)
    testset = MNIST('./data', train=False, transform=trans_img, download=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainset, testset, trainloader, testloader


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),   # in_channels, out_channels, kernal size
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),         # 32*14*14\
            nn.Dropout2d(0.6),

            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),      # 64*7*7
            nn.Dropout2d(0.5),
        )
        self.dis_layer = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024,1),
            nn.Sigmoid(),
        )
        self.classifier_layer = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256,10),
        )

    def forward(self, x):
        x = self.cov(x)
        x = x.view(x.size(0), -1)
        y = self.dis_layer(x)
        z = self.classifier_layer(x)
        z = F.softmax(z, dim=1)
        return y , z


class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # 1*56*56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.gen = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),

            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),

            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.gen(x)
        return x


if __name__ == "__main__":
    num_img = 100    # batch size
    z_dimension = 110
    loss1 = nn.BCELoss(reduction='mean') # loss1=BCE
    loss2 = nn.CrossEntropyLoss()  # loss2=CrossEntropy
    D = discriminator()
    G = generator(z_dimension, 3136)  # 1*56*56
    trainset, testset, trainloader, testloader = loadmnist(num_img)  # data
    D = D.cuda()
    G = G.cuda()
    d_optimizer = optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = optim.Adam(G.parameters(), lr=0.0003)
    '''
    交替训练的方式训练网络
    先训练判别器网络D再训练生成器网络G
    不同网络的训练次数是超参数
    也可以两个网络训练相同的次数，
    这样就可以不用分别训练两个网络
    '''
    count = 0
    epoch = 119
    gepoch = 2   # D和G轮流训练

    # 判别器D的训练,固定G的参数
    for i in tqdm.tqdm(range(epoch)):
        for (img, label) in trainloader:
            labels_onehot = np.zeros((num_img, 10))                # 100个label  100*10
            labels_onehot[np.arange(num_img), label.numpy()] = 1   # 相应标签位置置1 各行的任意列（0~9任意标签）
            #             img=img.view(num_img,-1)
            #             img=np.concatenate((img.numpy(),labels_onehot))
            #             img=torch.from_numpy(img)
            # 训练生成器的时候，由于生成网络的输入向量z_dimension = 110维，
            # 而且是100维随机向量和10维真实图片标签拼接，需要做相应的拼接操作
            img = Variable(img).cuda()

            # 类标签
            real_cls_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()  # 真的类别label相应为1  100*10
            fake_cls_label = Variable(torch.zeros(num_img, 10)).cuda()  # 假的类别label全为0   100*10

            # 真假标签
            real_dis_label = Variable(torch.ones(num_img,1).float()).cuda()   # 100*1
            fake_dis_label = Variable(torch.zeros(num_img,1).float()).cuda()  # 100*1

            # 真图片的损失
            real_dis_out , real_cls_out = D(img)  # 真图片送入判别器D  得到真假输出 100*1 和分类输出100*10

            # 真图片的真假损失
            input1 = real_dis_out.cuda()
            target1 = real_dis_label.cuda()
            d_loss_real_dis = loss1(input1,target1)        # 真图片的真假loss

            # 真图片的分类损失
            input_a = real_cls_out.cuda()
            target_a = label.cuda()
            d_loss_real_cls = loss2(input_a, target_a)

            # 得分
            real_dis_scores = real_dis_out  # 真图片真假得分 100*1
            real_cls_scores = real_cls_out  # 真图片分类得分 100*10

            # 假图片的损失
            z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 随机生成带标签的向量 100*110
            fake_img = G(z)  # 将向量放入生成网络G生成一张图片
            fake_dis_out , fake_cls_out = D(fake_img)  # 假图片送入判别器D  得到真假输出100*1 和 分类输出100*10
            # 假图片的真假损失
            input2 = fake_dis_out.cuda()
            target2 = fake_dis_label.cuda()
            d_loss_fake_dis = loss1(input2, target2)      # 假图片的真假loss    注意鉴别器训练

            # 假图片的分类损失
            input_b = fake_cls_out.cuda()
            target_b = label.cuda()
            d_loss_fake_cls = loss2(input_b, target_b)

            # 得分
            fake_dis_scores = fake_dis_out  # 假图片真假得分 100*1
            fake_cls_scores = fake_cls_out  # 假图片分类得分 100*10

            # D 反向传播与参数更新
            d_loss = d_loss_real_dis + d_loss_real_cls + d_loss_fake_dis + d_loss_fake_cls
            d_optimizer.zero_grad()  # 判别器D的梯度归零
            d_loss.backward()  # 反向传播
            d_optimizer.step()  # 更新判别器D参数

            # 生成器G的训练
            for j in range(gepoch):
                z = torch.randn(num_img, 100)  # 随机生成向量
                z = np.concatenate((z.numpy(), labels_onehot), axis=1)  # 100*110
                z = Variable(torch.from_numpy(z).float()).cuda()

                fake_img = G(z)  # 将向量放入生成网络G生成一张图片
                dis_out , cls_out = D(fake_img)  # 经过判别器得到结果 假图片真假输出100*1 和 假图片分类输出100*10
                # 生成器真假损失
                input3 = dis_out.cuda()
                target3 = real_dis_label.cuda()              # 假图片真假得分 与 1的损失  使生成器越来越真
                g_dis_loss = loss1(input3, target3)     # 得到假图片与真实标签的loss

                #生成器分类损失
                input_c = cls_out.cuda()
                target_c = label.cuda()                       # 假图片分类得分 与 真label的损失  使生成器越来越真
                g_cls_loss = loss2(input_c, target_c)         # 得到假图片与真实类标的loss

                # 反向传播与参数更新
                g_loss = g_dis_loss + g_cls_loss
                g_optimizer.zero_grad()  # 生成器G的梯度归零
                g_loss.backward()  # 反向传播
                g_optimizer.step()  # 更新生成器G参数
                temp = real_cls_label

        # 模型保存（10 epoch）
        if (i % 10 == 0) and (i != 0):
            print(i)
            torch.save(G.state_dict(), r'./ACGAN/Generator_cuda_%d.pkl' % i)
            torch.save(D.state_dict(), r'./ACGAN/Discriminator_cuda_%d.pkl' % i)
            save_model(G, r'./ACGAN/Generator_cpu_%d.pkl' % i)  # 保存为CPU中可以打开的模型
            save_model(D, r'./ACGAN/Discriminator_cpu_%d.pkl' % i)  # 保存为CPU中可以打开的模型

        # 可视化与保存
        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
              'D real_dis: {:.6f}, D real_cls: {:.6f}, D fake_dis: {:.6f},'
              ' D fake_cls: {:.6f}, G_loss_dis: {:.6f}, G_loss_cls: {:.6f}'.format(
            i, epoch, d_loss.data, g_loss.data,
            d_loss_real_dis.data, d_loss_real_cls.data, d_loss_fake_dis.data, d_loss_fake_cls.data, g_dis_loss.data, g_cls_loss.data))
        temp = temp.to('cpu')
        # 在输出图片的同时输出期望的类标签
        _, x = torch.max(temp, 1)  # 返回值有两个，第一个是每行的最大值，第二个是每行最大值的列标号 1*100
        x = x.numpy()
        print(x[[5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95]])  # 显示16张子图
        showimg(fake_img, count)
        plt.savefig(r'E:\Anaconda\Python projects\gan - 副本\ACGAN\images\Epoch_%d.jpg' % i, dpi=300)
        plt.show()
        plt.close()
        count += 1