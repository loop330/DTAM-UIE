import argparse
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# settings

def psnr_and_ssim(choice):

    if choice=='Train_UIEB':
        im_path = '../root/autodl-tmp/result/UIEB/train_output/image'
        re_path = '../root/autodl-tmp/data/UIEB/train/reference/'
        avg_psnr = 0
        avg_ssim = 0
        n = 0
    if choice == 'Val_UIEB':
        im_path = './result/UIEB/val_output_mid'
        re_path = './data/UIEB/val/reference/'
        avg_psnr = 0
        avg_ssim = 0
        n =0
    if choice == 'Test':
        im_path = './result/UIEB/test_output'
        re_path = './data/UIEB/test/reference/'
        avg_psnr = 0
        avg_ssim = 0
        n =0
    if choice == 'Train_UFO':
        im_path = '../root/autodl-tmp/result/UFO/train_output/image'
        re_path = '../root/autodl-tmp/data/UFO_120/train/reference/'
        avg_psnr = 0
        avg_ssim = 0
        n = 0
    if choice == 'Val_UFO':
        im_path = './result/UFO/val_output'
        re_path = './data/UFO_120/val/reference/'
        avg_psnr = 0
        avg_ssim = 0
        n = 0
        # for filename in os.listdir(im_path):
#     n = n + 1
#     im1  = cv2.imread(im_path+"/"+filename)
#     im2 = cv2.imread(re_path+"/"+filename)
#     (h,w,c) = im2.shape
    if choice =="others":
        im_path = './study_data/LSUI/LANET'
        re_path = './study_data/LSUI/gt/'
        avg_psnr = 0
        avg_ssim = 0
        n = 0
    for filename in os.listdir(im_path):

        n = n + 1
        im1 = cv2.imread(im_path + "/" + filename)
        im2 = cv2.imread(re_path + "/" + filename)

        im1 = cv2.resize(im1 ,(256,256))
        im2 = cv2.resize(im2 ,(256, 256))  # reference size

        score_psnr = psnr(im1, im2)
        score_ssim = ssim(im1, im2, channel_axis=-1)

        avg_psnr += score_psnr
        avg_ssim += score_ssim

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    return avg_psnr,avg_ssim

