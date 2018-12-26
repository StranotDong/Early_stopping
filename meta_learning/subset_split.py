import csv
import random
import os
from shutil import copyfile

# read the meta csv file
meta_file = 'meta_data.csv'
meta_list = []
with open(meta_file) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if row['dataset'] == 'ImageNet':
            dict_ = {}
            dict_['No.'] = i
            for key in row:
                dict_[key] = row[key]
            meta_list.append(dict_)
# print(meta_list)

# get all the labels
dataset_root = 'whole_dataset/raw_data/train'
labels = []
for _, dirnames, _  in os.walk(dataset_root):
    labels.append(dirnames)
labels = labels[0]

# split dataset
output_root = 'sub_datasets'
for co, dict_ in enumerate(meta_list):
    print('{} out of {}'.format(co+1, len(meta_list)))

    # train_size = float(dict_['train_data_size'])
    # ratio = float(dict_['train_val_ratio'])
    # val_size = train_size // ratio
    # total_size = train_size + val_size
    num_classes = float(dict_['num_classes'])
    # each_size = total_size // num_classes
    # each_train_size = int(each_size // (ratio + 1)*ratio)
    # remain_size = total_size % num_classes + each_size
    # each_remain_size = int(remain_size // (ratio + 1)*ratio)

    ## get the subset
    ran_labels = random.sample(labels, int(num_classes))
    subset = {}
    for i, label in enumerate(ran_labels):
        images = []
        # print(dataset_root + '/' + label)
        for _, _, filenames in os.walk(dataset_root + '/' + label):
            images.append(filenames)

        num_images = len(images[0])
        each_size = int(round(num_images * float(dict_['used_data_ratio'])))
        ratio = float(dict_['train_val_ratio'])
        each_train_size = int(round(each_size / (ratio + 1)*ratio))

        ran_images = random.sample(images[0], int(each_size))
        train = ran_images[:each_train_size]
        val = ran_images[each_train_size:]
        subset[label] = {'train':train, 'val':val}
    # images = []
    # for _, _, filenames in os.walk(dataset_root + '/' + ran_labels[-1]):
    #     images.append(filenames)
    # ran_images = random.sample(images[0], int(remain_size))
    # train = ran_images[:each_remain_size]
    # val = ran_images[each_remain_size:]
    # subset[ran_labels[-1]] = {'train':train, 'val':val}

    ## copy the related images to proper places
    folder = str(co)
    path = output_root + '/' + folder
    if not os.path.exists(path):
        os.makedirs(path)
    train_path = path + '/train'
    val_path = path + '/validation'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    for label in subset:
        src_dir = dataset_root + '/' + label
        train_image_dir = train_path + '/'+ label
        if not os.path.exists(train_image_dir):
            os.makedirs(train_image_dir)
        for img in subset[label]['train']:
            copyfile(src_dir+'/'+img, train_image_dir+'/'+img)
        val_image_dir = val_path + '/'+ label
        if not os.path.exists(val_image_dir):
            os.makedirs(val_image_dir)
        for img in subset[label]['val']:
            copyfile(src_dir+'/'+img, val_image_dir+'/'+img)

# print(subset)
# sum_ = 0
# for i in subset:
#     sum_ += len(subset[i]['train']) + len(subset[i]['val'])
# print(sum_)

