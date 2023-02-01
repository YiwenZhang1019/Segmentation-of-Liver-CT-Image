# -*- coding: utf-8 -*-
import os
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
from LungDataset import LungDataset,LiverDataset
from models.common_tools import set_seed
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from models.networks_cenet import CE_Net_backbone_DAC_without_atrous,CE_Net_
from loss.FOCALLOSS import FocalLossV1
from models.UNet import UNet,bishe,UNet1
from models.UNet_2Plus import UNet_2Plus
from loss.mixloss import mixloss
from loss.mixxloss import MS_SSIM



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x_transforms = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize((240,240)),
    transforms.ToTensor(),
])

set_seed()  # 设置随机种子


def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return:
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def recall(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)

def pre(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (output.sum() + smooth)


if __name__ == "__main__":

    # config
    LR = 0.01
    BATCH_SIZE =2
    max_epoch =60 # 400
    start_epoch = 0
    lr_step =100
    val_interval = 2
    checkpoint_interval = 5
    vis_num = 10
    mask_thres = 0.5
    data_dir = os.path.abspath(os.path.join(BASE_DIR, "chao"))
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "val")

    # step 1
    train_set = LiverDataset(train_dir,transform=x_transforms,target_transform=y_transforms)
    valid_set = LiverDataset(valid_dir,transform=x_transforms,target_transform=y_transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True, drop_last=False)

    # step 2
    net =UNet_2Plus(in_channels=3, n_classes=1 )   # init_features is 64 in stander uent
    net.to(device)

    # step 3
    # loss_fn = mixloss()
    # loss_fn=nn.BCELoss()
    loss_fn=FocalLossV1()
    # step 4
    optimizer = optim.SGD(net.parameters(), lr=LR)
    # optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5)
    # step 5
    train_curve = list()
    valid_curve = list()
    train_dice_curve = list()
    train_recall_curve=list()
    train_pre_curve=list()
    valid_dice_curve = list()
    valid_recall_curve=list()
    valid_pre_curve = list()
    for epoch in range(start_epoch, max_epoch):

        train_loss_total = 0.
        train_dice_total = 0.

        net.train()
        for iter, (inputs, labels) in enumerate(train_loader):

            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            # print
            train_curve.append(loss.item())
            train_dice = compute_dice(outputs.ge(mask_thres).cpu().data.numpy(), labels.cpu())
            train_dice_curve.append(train_dice)
            train_recall=recall(outputs.ge(mask_thres).cpu().data.numpy(), labels.cpu())
            train_recall_curve.append(train_recall)
            train_pre = pre(outputs.ge(mask_thres).cpu().data.numpy(), labels.cpu())
            train_pre_curve.append(train_pre)

            train_loss_total += loss.item()

            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] running_loss: {:.4f}, mean_loss: {:.4f} "
                  "running_dice: {:.4f} recall:{:.4f} pre:{:.4f} lr:{}".format(epoch, max_epoch, iter + 1, len(train_loader), loss.item(),
                                    train_loss_total/(iter+1), train_dice,train_recall ,train_pre,scheduler.get_lr()))

        scheduler.step()

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "./checkpoint_{}_epoch.pkl".format(epoch)
            torch.save(checkpoint, path_checkpoint)

        # validate the model
        if (epoch+1) % val_interval == 0:

            net.eval()
            valid_loss_total = 0.
            valid_dice_total = 0.
            valid_recall_total=0.
            valid_pre_total=0.
            num=0

            with torch.no_grad():
                for j, (inputs, labels) in enumerate(valid_loader):
                    if torch.cuda.is_available():
                        inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels )

                    valid_loss_total += loss.item()

                    valid_dice = compute_dice(outputs.ge(mask_thres).cpu().data, labels.cpu())
                    if 0.001<valid_dice<0.9999:
                        num+=1
                        valid_dice_total += valid_dice
                    valid_recall = recall(outputs.ge(mask_thres).cpu().data.numpy(), labels.cpu())
                    if 0.001<valid_recall<0.9999:
                        valid_recall_total += valid_recall
                    valid_pre = pre(outputs.ge(mask_thres).cpu().data.numpy(), labels.cpu())
                    if 0.001<valid_recall<0.9999:
                        valid_pre_total += valid_pre


                # valid_loss_mean = valid_loss_total/len(valid_loader)
                # valid_dice_mean = valid_dice_total/len(valid_loader)
                valid_loss_mean = valid_loss_total/len(valid_loader)
                valid_dice_mean = valid_dice_total/num
                valid_recall_mean=valid_recall_total/num
                valid_pre_mean = valid_pre_total/ num
                valid_curve.append(valid_loss_mean)
                valid_dice_curve.append(valid_dice_mean)


                print("Valid:\t Epoch[{:0>3}/{:0>3}] mean_loss: {:.4f} dice_mean: {:.4f} recall_mean:{:.4f} pre_mean:{:.4f}".format(
                    epoch, max_epoch, valid_loss_mean, valid_dice_mean,valid_recall_mean,valid_pre_mean))

    # 可视化
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            if idx > vis_num:
                break
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            pred = outputs.ge(mask_thres)

            mask_pred = outputs.ge(0.5).cpu().data.numpy().astype("uint8")

            img_hwc = inputs.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)).astype("uint8")
            plt.subplot(121).imshow(img_hwc)
            mask_pred_gray = mask_pred.squeeze(0) * 255
            plt.subplot(122).imshow(mask_pred_gray, cmap="gray")
            plt.show()
            plt.pause(0.5)
            plt.close()

    # plot curve
    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(
        valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.title("Plot in {} epochs".format(max_epoch))
    plt.show()

    # dice curve
    train_x = range(len(train_dice_curve))
    train_y = train_dice_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(
        valid_dice_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_dice_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('dice value')
    plt.xlabel('Iteration')
    plt.title("Plot in {} epochs".format(max_epoch))
    plt.show()
    torch.cuda.empty_cache()















