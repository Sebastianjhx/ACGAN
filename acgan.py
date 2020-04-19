import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
import matplotlib.gridspec as gridspec
from PIL import Image


def save_model(model, filename):  # 保存为CPU中可以打开的模型
    state = model.state_dict()
    x = state.copy()
    for key in x:
        x[key] = x[key].clone().cpu()
    torch.save(x, filename)
# state_dict作为python的字典对象将每一层的参数映射成tensor张量


def showimg(images):
    images = images.to('cpu')
    images = images.detach().numpy()
    images = images[[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]]
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    grid_length = int(np.ceil(np.sqrt(images.shape[0])))    # 3
    plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(grid_length, grid_length, wspace=0, hspace=0)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.transpose(1,2,0))
        plt.axis('off')
        plt.tight_layout()
    #     plt.tight_layout()
    # plt.savefig(r'./CGAN/images/%d.png' % count, bbox_inches='tight')



def loadcap(batch_size):  # 图片的大小是64*64
    trans_img = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    trainset = ImageFolder('./data/train/',  transform=trans_img)
    testset = ImageFolder('./data/test/',  transform=trans_img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    return trainset, testset, trainloader, testloader


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.cov  = nn.Sequential(
                    nn.Conv2d(in_channels = 3,              # 64*32*32
                             out_channels = 64,
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Dropout(0.5, inplace=False),

                    nn.Conv2d(in_channels = 64,            # 128*16*16
                             out_channels = 128,
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Dropout(0.5, inplace=False),

                    nn.Conv2d(in_channels = 128,           # 256*8*8
                             out_channels = 256,
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Dropout(0.5, inplace=False),

                    nn.Conv2d(in_channels = 256,            # 512*4*4
                             out_channels = 512,
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Dropout(0.5, inplace=False)
                    )
        self.dis_layer = nn.Sequential(
            nn.Linear(512*4*4, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512,1),
            nn.Sigmoid(),
        )
        self.classifier_layer = nn.Sequential(
            nn.Linear(512*4*4, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 15),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cov(x)
        x = x.view(-1, 512*4*4)
        y = self.dis_layer(x)
        z = self.classifier_layer(x)
        return y , z


class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # 512*4*4
        self.br = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,          # 256*8*8
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256,           # 128*16*16
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128,           # 64*32*32
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64,          # 3*64*64
                               out_channels=3,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)

        x = self.gen(x)
        return x


if __name__ == "__main__":
    num_img = 100  # batch size
    z_dimension = 115
    loss1 = nn.BCELoss(reduction='mean')  # loss1=BCE
    loss2 = nn.NLLLoss()  # loss2=NLLLoss
    D = discriminator()
    G = generator(z_dimension, 8192)  # 512*4*4
    # D.load_state_dict(torch.load(r'./ACGAN/Discriminator_cuda_190.pkl'))
    # G.load_state_dict(torch.load(r'./ACGAN/Generator_cuda_190.pkl'))
    trainset, testset, trainloader, testloader = loadcap(num_img)  # data
    D = D.cuda()
    G = G.cuda()
    d_optimizer = optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = optim.Adam(G.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=1e-3)

    '''
    交替训练的方式训练网络
    先训练判别器网络D再训练生成器网络G
    不同网络的训练次数是超参数
    也可以两个网络训练相同的次数，
    这样就可以不用分别训练两个网络
    '''
    count = 0
    epoch = 200
    gepoch = 1  # D和G轮流训练

    # 判别器D的训练,固定G的参数:
    for i in tqdm.tqdm(range(epoch)):
        for (img, label) in trainloader:
            a = label.numpy()
            label_list = a.tolist()
            b = np.arange(num_img)
            b = b.tolist()
            labels_onehot = np.zeros((num_img, 15))  # 100个label  100*15
            labels_onehot[b, label_list] = 1  # 相应标签位置置1 各行的任意列（0~15任意标签）
            #             img=img.view(num_img,-1)
            #             img=np.concatenate((img.numpy(),labels_onehot))
            #             img=torch.from_numpy(img)
            # 训练生成器的时候，由于生成网络的输入向量z_dimension = 115维
            # 而且是100维随机向量和10维真实图片标签拼接，需要做相应的拼接操作
            img = Variable(img).cuda()

            d_optimizer.zero_grad()  # 判别器D的梯度归零

            # 类标签
            real_cls_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()  # 真的类别label相应为1  100*15
            fake_cls_label = Variable(torch.zeros(num_img, 15)).cuda()  # 假的类别label全为0   100*15

            # 真假标签
            real_dis_label = Variable(torch.ones(num_img, 1).float()).cuda()  # 100*1
            fake_dis_label = Variable(torch.zeros(num_img, 1).float()).cuda()  # 100*1

            # 真图片的损失
            real_dis_out, real_cls_out = D(img)  # 真图片送入判别器D  得到真假输出 100*1 和分类输出100*15

            # 真图片的真假损失
            input1 = real_dis_out.cuda()
            target1 = real_dis_label.cuda()
            d_loss_real_dis = loss1(input1, target1)  # 真图片的真假loss

            # 真图片的分类损失
            input_a = real_cls_out.cuda()
            target_a = label.cuda()
            d_loss_real_cls = loss2(input_a, target_a)

            # 得分
            d_loss_real = d_loss_real_dis + d_loss_real_cls
            d_loss_real.backward()

            # 假图片的损失
            z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 随机生成带标签的向量 100*115
            fake_img = G(z)  # 将向量放入生成网络G生成一张图片
            fake_dis_out, fake_cls_out = D(fake_img)  # 假图片送入判别器D  得到真假输出100*1 和 分类输出100*15

            # 假图片的真假损失
            input2 = fake_dis_out.cuda()
            target2 = fake_dis_label.cuda()
            d_loss_fake_dis = loss1(input2, target2)  # 假图片的真假loss    注意鉴别器训练

            # 假图片的分类损失
            input_b = fake_cls_out.cuda()
            target_b = label.cuda()
            d_loss_fake_cls = loss2(input_b, target_b)

            # 得分
            d_loss_fake = d_loss_fake_dis + d_loss_fake_cls
            d_loss_fake.backward()

            # D 反向传播与参数更新
            d_loss = d_loss_fake + d_loss_real
            d_optimizer.step()  # 更新判别器D参数

            # 生成器G的训练
            for j in range(gepoch):
                z = torch.randn(num_img, 100)  # 随机生成向量
                z = np.concatenate((z.numpy(), labels_onehot), axis=1)  # 100*110
                z = Variable(torch.from_numpy(z).float()).cuda()

                g_optimizer.zero_grad()  # 生成器G的梯度归零

                fake_img = G(z)  # 将向量放入生成网络G生成一张图片  num_img*3*64*64
                dis_out, cls_out = D(fake_img)  # 经过判别器得到结果 假图片真假输出100*1 和 假图片分类输出100*15
                # 生成器真假损失
                input3 = dis_out.cuda()
                target3 = real_dis_label.cuda()  # 假图片真假得分 与 1的损失  使生成器越来越真
                g_dis_loss = loss1(input3, target3)  # 得到假图片与真实标签的loss

                # 生成器分类损失
                input_c = cls_out.cuda()
                target_c = label.cuda()  # 假图片分类得分 与 真label的损失  使生成器越来越真
                g_cls_loss = loss2(input_c, target_c)  # 得到假图片与真实类标的loss

                # 反向传播与参数更新
                g_loss = g_dis_loss + g_cls_loss
                g_loss.backward()  # 反向传播
                g_optimizer.step()  # 更新生成器G参数
                temp = real_cls_label

        # 模型保存（10 epoch）
        if (i % 10 == 0) and (i != 0):
            print(i)
            torch.save(G.state_dict(), r'./ACGAN/Generator_cuda_%d.pkl' % i)
            torch.save(D.state_dict(), r'./ACGAN/Discriminator_cuda_%d.pkl' % i)
            # save_model(G, r'./ACGAN/Generator_cpu_%d.pkl' % i)  # 保存为CPU中可以打开的模型
            # save_model(D, r'./ACGAN/Discriminator_cpu_%d.pkl' % i)  # 保存为CPU中可以打开的模型

            # 可视化与保存
        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
              'D real_dis: {:.6f}, D real_cls: {:.6f}, D fake_dis: {:.6f},'
              ' D fake_cls: {:.6f}, G_loss_dis: {:.6f}, G_loss_cls: {:.6f}'.format(
            i, epoch, d_loss.data, g_loss.data,
            d_loss_real_dis.data, d_loss_real_cls.data,
            d_loss_fake_dis.data, d_loss_fake_cls.data,
            g_dis_loss.data, g_cls_loss.data))
        temp = temp.to('cpu')
        # 在输出图片的同时输出期望的类标签
        _, x = torch.max(temp, 1)  # 返回值有两个，第一个是每行的最大值，第二个是每行最大值的列标号 1*100
        x = x.numpy()
        print(x[[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]])  # 显示16张子图
        showimg(fake_img)
        plt.savefig(r'E:\Anaconda\Python projects\gan - 副本\ACGAN\images\Epoch_%d.jpg' % i, dpi=300)
        plt.show()
        plt.close()