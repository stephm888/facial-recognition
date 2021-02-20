import os
import PIL.Image
import h5py
import numpy as np
from tqdm import tqdm

type_number = {'train': [1, 2, 3, 4, 5, 6, 7, 8],
               'test': [9, 10]}
txt_folder = 'data/lfw_funneled'
img_folder='data/lfw_funneled'
shape=(224, 224)

def is_same_people(name_1, name_2):
    people_name_1 = os.path.split(name_1)[0]
    people_name_2 = os.path.split(name_2)[0]

    return int(people_name_1 == people_name_2)


def np_img_by_name(name):
    img = PIL.Image.open(name)
    img = img.resize(shape)
    img = np.array(img)
    return img

def save_img_to_h5(type='train'):
    pair_img_1 = []
    pair_img_2 = []
    labels = []
    for i in tqdm(type_number[type]):
        txt_name = txt_folder + '/pairs_{:02}.txt'.format(i)
        names = open(txt_name).readlines()
        pair_num = len(names) // 5
        for j in range(pair_num):
            name_1 = os.path.join(img_folder, names[j * 5 + 0].strip())
            name_2 = os.path.join(img_folder, names[j * 5 + 1].strip())
            name_3 = os.path.join(img_folder, names[j * 5 + 2].strip())
            name_4 = os.path.join(img_folder, names[j * 5 + 3].strip())

            pair_img_1.append(np_img_by_name(name_1))
            pair_img_2.append(np_img_by_name(name_2))
            pair_img_1.append(np_img_by_name(name_3))
            pair_img_2.append(np_img_by_name(name_4))
            labels.append(is_same_people(name_1, name_2))
            labels.append(is_same_people(name_3, name_4))
    np_pair_img_1 = np.stack(pair_img_1)
    np_pair_img_2 = np.stack(pair_img_2)
    np_label = np.array(labels)
    print(np_pair_img_1.shape,np_pair_img_2.shape,np_label.shape)
    print(np_label[:20])

    fname = "data/h5/%s.h5" % (type)
    with h5py.File(fname) as h:
        h.create_dataset(type+'_1', data=np_pair_img_1)
        h.create_dataset(type+'_2', data=np_pair_img_2)
        h.create_dataset("label", data=np_label)
    print(fname)

if __name__ == '__main__':
    save_img_to_h5('train') #4800
    save_img_to_h5('test') #1200
