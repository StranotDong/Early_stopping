import random
import csv
import json

# about data set
dataset = ['ImageNet', 'OpenImages']
# train_data_size = [5e3,1e4,2e4,3e4,4e4,5e4,6e4,7e4,8e4,9e4,10e4]
train_val_ratio = [5,10]
image_size = [32, 64, 128, 224, 256]
num_classes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
used_data_ratio = [0.5, 0.8, 1.0]
# about NN structure
lr = [0.01, 0.05, 0.1]
bs = [64, 96, 128, 160]
resnet_size = [50, 56, 62, 68]
kernel_size = [3,5,7]
num_filters = [16,32,48,64,80,96]

meta_feat_dict = {'dataset':dataset,
                  # 'train_data_size':train_data_size,
                  'train_val_ratio':train_val_ratio,
                  'image_size':image_size,
                  'num_classes':num_classes,
                  'used_data_ratio':used_data_ratio,
                  'learning_rate':lr,
                  'batch_size':bs,
                  'resnet_size':resnet_size,
                  'kernel_size':kernel_size,
                  'num_filters':num_filters}

num_metadata = 50

rst_list = []
for i in range(num_metadata):
    feat_dict = {}

    for k in meta_feat_dict:
        l = len(meta_feat_dict[k])
        index = random.randrange(l)
        feat_dict[k] = meta_feat_dict[k][index]

    rst_list.append(feat_dict)

print(rst_list)

with open('meta_data.csv', 'w+') as f:
    fieldname=meta_feat_dict.keys()
    writer = csv.DictWriter(f,fieldname )
    writer.writeheader()
    for rst in rst_list:
        writer.writerow(rst)
