import torch
import torchvision
import os
from PIL import Image
from data_aug import augment
class UIEB_dataset_train(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path = "./data_rename_augment/UIEB/train/image/"
        self.reference_path = "./data_rename_augment/UIEB/train/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path+self.image_name_list[item])
        reference = Image.open(self.reference_path+self.reference_name_list[item])
        image = image.resize([256,256])
        reference = reference.resize([256,256])
        # image,reference = augment(image,reference)
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)
        return image_tensor,reference_tensor

class dataset_UIEB_test(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path ="./data/UIEB/test/image/"
        self.reference_path = "./data/UIEB/test/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path + self.image_name_list[item])
        reference = Image.open(self.reference_path + self.reference_name_list[item])
        image = image.resize([256, 256])
        reference = reference.resize([256, 256])
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)
        return image_tensor,reference_tensor

class UIEB_dataset_val(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path = "./data/UIEB/val/image/"
        self.reference_path = "./data/UIEB/val/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path+self.image_name_list[item])
        reference = Image.open(self.reference_path+self.reference_name_list[item])
        image = image.resize([256,256])
        reference = reference.resize([256,256])
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)
        return image_tensor,reference_tensor,self.image_name_list[item]

class UFO_dataset_train(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path = "./data/UFO_120/train/image/"
        self.reference_path = "./data/UFO_120/train/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path+self.image_name_list[item])
        reference = Image.open(self.reference_path+self.reference_name_list[item])
        image = image.resize([256,256])
        reference = reference.resize([256,256])
        # image,reference = augment(image,reference)
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)
        return image_tensor,reference_tensor
class UFO_dataset_val(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path = "./data/UFO_120/val/image/"
        self.reference_path = "./data/UFO_120/val/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path+self.image_name_list[item])
        reference = Image.open(self.reference_path+self.reference_name_list[item])
        image = image.resize([256,256])
        reference = reference.resize([256,256])
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)

        return image_tensor,reference_tensor,self.image_name_list[item]
class EUVP_SCENE_dataset_train(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path = "./data/EUVP/scene/train/image/"
        self.reference_path = "./data/EUVP/scene/train/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path+self.image_name_list[item])
        reference = Image.open(self.reference_path+self.reference_name_list[item])
        image = image.resize([256,256])
        reference = reference.resize([256,256])
        # image,reference = augment(image,reference)
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)
        return image_tensor,reference_tensor
class EUVP_SCENE_dataset_val(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path = "./data/EUVP/scene/val/image/"
        self.reference_path = "./data/EUVP/scene/val/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path+self.image_name_list[item])
        reference = Image.open(self.reference_path+self.reference_name_list[item])
        image = image.resize([256,256])
        reference = reference.resize([256,256])
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)

        return image_tensor,reference_tensor,self.image_name_list[item]
class LSUI_dataset_train(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path = "./data/LSUI/train/image/"
        self.reference_path = "./data/LSUI/train/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path+self.image_name_list[item])
        reference = Image.open(self.reference_path+self.reference_name_list[item])
        image = image.resize([256,256])
        reference = reference.resize([256,256])
        # image,reference = augment(image,reference)
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)
        return image_tensor,reference_tensor
class LSUI_dataset_val(torch.utils.data.Dataset):
    def __init__(self):
        self.image_path = "./data/LSUI/val/image/"
        self.reference_path = "./data/LSUI/val/reference/"
        self.image_name_list = os.listdir(self.image_path)
        self.reference_name_list = os.listdir(self.reference_path)
    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self, item):
        image = Image.open(self.image_path+self.image_name_list[item])
        reference = Image.open(self.reference_path+self.reference_name_list[item])
        image = image.resize([256,256])
        reference = reference.resize([256,256])
        image_tensor = torchvision.transforms.ToTensor()(image)
        reference_tensor = torchvision.transforms.ToTensor()(reference)

        return image_tensor,reference_tensor,self.image_name_list[item]