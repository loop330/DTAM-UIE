import shutil
import os
def move_reslt(data_name):
    if data_name == "UIEB":
        file_source = "./result/UIEB/val_output_mid/"
        file_destination = "./result/UIEB/val_output/"

        get_files = os.listdir(file_source)
        get_files2 = os.listdir(file_destination)
        for g in get_files2:
            os.remove(file_destination+g)
        for g in get_files:
            shutil.move(file_source+g, file_destination)
    if data_name =="EUVP_scene":
        file_source = "./result/EUVP/scene/val_output_mid/"
        file_destination = "./result/EUVP/scene/val_output/"

        get_files = os.listdir(file_source)
        get_files2 = os.listdir(file_destination)
        for g in get_files2:
            os.remove(file_destination + g)
        for g in get_files:
            shutil.move(file_source + g, file_destination)
    if data_name =="LSUI":
        file_source = "./result/LSUI/val_output_mid/"
        file_destination = "./result/LSUI/val_output/"

        get_files = os.listdir(file_source)
        get_files2 = os.listdir(file_destination)
        for g in get_files2:
            os.remove(file_destination + g)
        for g in get_files:
            shutil.move(file_source + g, file_destination)
