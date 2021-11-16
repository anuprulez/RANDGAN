import h5py
import numpy as np
from PIL import Image
import os


base_path = "/data/users/backup/anup/randgan/create_randgan_datasets/data/"
#base_path = "/home/anupkumar/randgan/RANDGAN/RANDGAN/"
train_path = base_path + "train/"
test_path = base_path + "test/"
test_pred_path = base_path + "test_pred/"
tr_h5 = "train_images_negative_txt_file.h5"
te_h5 = "te_images_all_txt_file.h5"
te_pred_h5 = "te_pred_images_all_txt_file.h5"


def read_text_file(file_path):
    with open(file_path, 'r') as txt_f:
        f_content = txt_f.read()
        file_names = [fi for fi in f_content.split("\n") if fi != '']
        return file_names

def convert_hdf5_train(files, image_path, output_path, key):
    l_images = list()
    ctr = 0
    hf_file = h5py.File(output_path, 'w')
    for img_file in files:
        s_name = img_file.split(' ')
        # take only COVID negative images
        image_file_name = s_name[-3]
        if s_name[-2] == "negative":
            try:
                img = Image.open(image_path + image_file_name).convert('L')
                img = np.asarray(img.resize((256, 256)))
                l_images.append(img)
                ctr += 1
                print("Image processed: {}".format(str(ctr)))
            except:
                continue
    l_images = np.array(l_images)
    print(l_images.shape)
    hf_file.create_dataset(key, data=l_images)
    hf_file.close()


def convert_hdf5_test(files, image_path, output_path, key):
    l_images = list()
    ctr = 0
    hf_file = h5py.File(output_path, 'w')
    for img_file in files:
        s_name = img_file.split(' ')
        image_file_name = s_name[-3]
        try:
            img = Image.open(image_path + image_file_name).convert('L')
            img = np.asarray(img.resize((256, 256)))
            l_images.append(img)
            ctr += 1
            print("Image processed: {}".format(str(ctr)))
        except:
            continue
    l_images = np.array(l_images)
    print(l_images.shape)
    hf_file.create_dataset(key, data=l_images)
    hf_file.close()


print("Saving training images...")
convert_hdf5_train(read_text_file("train_split.txt"), train_path, tr_h5, "tr_images")

print("Saving test images...")
convert_hdf5_test(read_text_file("test_split.txt"), test_path, te_h5, "te_images")

print("Saving test pred images...")
convert_hdf5_test(read_text_file("test_split.txt"), test_pred_path, te_pred_h5, "te_pred_images")
