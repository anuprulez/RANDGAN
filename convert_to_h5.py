import h5py
import numpy as np
from PIL import Image
import os


base_path = "/data/users/backup/anup/randgan/create_randgan_datasets/data/"
train_path = base_path + "train/"
test_path = base_path + "test/"
test_pred_path = base_path + "test_pred/"
tr_h5 = "train_images.h5"
te_h5 = "te_images.h5"
te_pred_h5 = "te_pred_images.h5"


def convert_hdf5(image_path, output_path, key):
    l_images = list()
    ctr = 0
    hf_file = h5py.File(output_path, 'w')
    for img_file in os.listdir(image_path):
        try:
            img = Image.open(image_path + img_file).convert('L')
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
convert_hdf5(train_path, train_path + tr_h5, "tr_images")

print("Saving test images...")
convert_hdf5(test_path, test_path + te_h5, "te_images")

print("Saving test pred images...")
convert_hdf5(test_pred_path, test_pred_path + te_pred_h5, "te_pred_images")
