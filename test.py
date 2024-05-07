import torch
from data import dataset_UIEB_test
from torchvision import transforms
import torch.utils.data as Data
from swin_unet_transformer import SwinTransformerSys
from psnr_ssim import psnr_and_ssim
def test_data_process():
    test_data = dataset_UIEB_test()
    test_data_loader = Data.DataLoader(test_data,batch_size=1,shuffle=True)
    return test_data_loader
def test(model,test_data_loader):

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    with torch.no_grad():
        for image,reference in test_data_loader:
            image = image.to(device)
            reference = reference.to(device)
            model.eval()
            output = model(image)

    psnr,ssim = psnr_and_ssim(choice="Test")