import copy
import datetime
import time
import pandas as pd
import torch
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from data import UIEB_dataset_train,UIEB_dataset_val
from data import UFO_dataset_train,UFO_dataset_val
from data import EUVP_SCENE_dataset_train,EUVP_SCENE_dataset_val
from data import LSUI_dataset_train,LSUI_dataset_val
from loss_function import  combinedloss
from psnr_ssim import psnr_and_ssim
import torchvision
from write_psnr_ssim import write_train_val_log
from ubiformer_cts import biformer_layer_unet
from tqdm import tqdm
from move_result import move_reslt
def train_val_data_process():
    train_data= UIEB_dataset_train()
    val_data = UIEB_dataset_val()
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=2,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data,batch_size = 1,shuffle=True)

    return train_dataloader,val_dataloader
def train_model_process(model,train_dataloader,val_dataloader,num_epoch,model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = datetime.date.today()
    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.8)
    #定义损失函数
    criterion = combinedloss()
    criterion = criterion.to(device)

    model = model.to(device)

    #复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    #初始化参数
    #最高psnr和ssim
    best_psnr = 0.0
    best_ssim = 0.0
    #训练集和验证集上的全部损失值
    train_loss_all = []
    val_loss_all = []

    #训练集和测试集的psnr和ssim
    val_psnr = 0.0
    val_ssim = 0.0
    #验证集全部的psnr和ssim
    val_psnr_all = []
    val_ssim_all = []
    #时间

    for epoch in range(num_epoch):
        log = []
        log.append(epoch+1)
        start_time = time.time()
        print("epcoh:{}/{}".format(epoch,num_epoch-1))
        print("-"*10)
        train_loss = 0.0
        val_loss = 0.0
        train_num = 0
        val_num = 0

        model.train()
        #对每一个mini-batch训练和计算
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for image,reference in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                image = image.to(device)
                reference = reference.to(device)
                output= model(image)
                loss = criterion(output,reference)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                #对损失函数累加
                train_loss = train_loss + loss.item() * image.shape[0]
                train_num = train_num + image.shape[0]

        train_loss_all.append(train_loss/train_num)
        log.append(train_loss/train_num)

        model.eval()
        for step,(image,reference,name) in enumerate(val_dataloader):
            names = str(name)
            names_list = names.split("'")
            save_image_name = names_list[1]

            image = image.to(device)
            reference = image.to(device)
            output= model(image)
            save_image_tensor = output[0].clone().detach().to(torch.device('cpu'))  # 到cpu
            torchvision.utils.save_image(save_image_tensor, "./result/UIEB/val_output_mid/"+save_image_name)
            loss = criterion(output,reference)
            val_loss = val_loss + loss.item() * image.shape[0]
            val_num = val_num + image.shape[0]
        val_loss_all.append(val_loss/val_num)
        val_psnr,val_ssim = psnr_and_ssim(choice="Val_UIEB")
        if(best_psnr<val_psnr ):
            best_psnr,best_ssim = val_psnr,val_ssim
            move_reslt("UIEB")
            torch.save(obj=model.state_dict(), f="./weight/best_model_UIEB.pth")
        val_psnr_all.append(val_psnr)
        val_ssim_all.append(val_ssim)
        log.append(val_loss/val_num)
        log.append(val_psnr)
        log.append(val_ssim)
        log.append(model_name)
        log.append(train_data)
        print("best_psnr:{:.4f},best_ssim:{:.4f}".format(best_psnr,best_ssim))
        print("train_loss:{:.4f}".format(train_loss/train_num))
        print("val_loss:{:.4f},val_psnr:{:.4f},val_ssim:{:.4f}".format(val_loss/val_num,val_psnr,val_ssim))
        end_time = time.time()
        epoch_need_time = end_time - start_time
        log.append(epoch_need_time)
        log.append("UIEB")
        print("epoch_time:{:.4f}s".format(epoch_need_time))
        write_train_val_log(log)
        print("log has saved")

    # if val_psnr>best_psnr or val_ssim>best_ssim:
    #     best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    torch.save(model.load_state_dict(best_model_wts), "./weight/best_model.pth")
    train_process = pd.DataFrame(data={"epoch":range(num_epoch),
                                    "train_loss_all":train_loss_all,
                                    "val_loss_all":val_loss_all,

                                    "val_psnr_all":val_psnr_all,
                                    "val_ssim_all":val_ssim_all,
    })
    return train_process
def matplot_psnr_ssim_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)#一行两列的第一张图
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,'bs-',label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

if __name__ == '__main__':
    model_name = "ubiformer_cts_cbam_last_process_cbam_improve2"
    model_restoration =  biformer_layer_unet()

    train_dataloader,val_dataloader = train_val_data_process()

    result = train_model_process(model_restoration, train_dataloader, val_dataloader,200,model_name=model_name)










