import csv
import os
def write_train_val_log(epoch_result_data):
    if os.path.exists("./train_log.csv"):
        with open("train_log.csv","a+",encoding="utf-8",newline="") as file:
            csv_writer=csv.writer(file)
            csv_writer.writerow(epoch_result_data)
            print("epcoch{}_result_has_saved".format(epoch_result_data[0]))
            file.close()
    else:
        with open("./train_log.csv","a+",encoding="utf-8",newline="") as file:
            project_save_name = ["epoch","train_loss","val_loss","val_psnr","val_ssim","model_name","date_time","epoch_time","data_name"]
            csv_write = csv.writer(file)
            csv_write.writerow(project_save_name)
            csv_write.writerow(epoch_result_data)
            print("epcoch{}_result_has_saved".format(epoch_result_data[0]))
            file.close()


