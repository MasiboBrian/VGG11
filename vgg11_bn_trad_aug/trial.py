import pickle
import numpy as np
from PIL import Image
import os
import random
random.seed(1) # set a seed so that the results are consistent

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def to_pickle(data, file):
    import pickle
    with open(file, 'wb') as fo:
        pickle.dump(data, fo, protocol=-1)


# trial_file = '/afs/inf.ed.ac.uk/user/s20/s2089883/MLP_CW/Coursework3/ml_cw3/Cluster/MonetBatch1.pickle'
# orig_file = '/afs/inf.ed.ac.uk/user/s20/s2089883/MLP_CW/Coursework3/ml_cw3/Cluster/CIFAR10_Data/cifar-10-batches-py/data_batch_1'
# read_dict = unpickle(trial_file)
# read_dict = unpickle(orig_file)


# i = read_dict[b'data'][0]
# print(i.shape)


# img = Image.open(img_path)
# img_np = np.asarray(img)
# for i in range(0,3):
# 	img_flat = np.append(img_flat, np.array(img_np[:,:,i].flatten()))

# img_flat = img_flat.astype('uint8')

# red = i[:(32*32)].reshape(32,32)
# green = i[(32*32):(32*32)*2].reshape(32,32)
# blue = i[(32*32)*2:(32*32)*3].reshape(32,32)

# rgb = np.dstack((red,green,blue))
# # i = i.reshape(32,32,-1)
# # print(i.shape)

# img = Image.fromarray(rgb)
# img.show()

# #flatten original image
# img_path = '/afs/inf.ed.ac.uk/user/s20/s2089883/MLP_CW/Coursework3/ml_cw3/Cluster/train_1000_7.png'

# img = Image.open(img_path)
# img.show()
# img_flat = np.array([])
# img_np = np.asarray(img)
# for i in range(0,3):
# 	img_flat = np.append(img_flat, np.array(img_np[:,:,i].flatten()))
# img_flat = img_flat.astype('uint8')

# #convert back to RGB and show
# red = img_flat[:(32*32)].reshape(32,32)
# green = img_flat[(32*32):(32*32)*2].reshape(32,32)
# blue = img_flat[(32*32)*2:(32*32)*3].reshape(32,32)

# rgb = np.dstack((red,green,blue))
# # i = i.reshape(32,32,-1)
# # print(i.shape)

# img = Image.fromarray(rgb)
# img.show()

#repeat for fake image
# img_path = '/afs/inf.ed.ac.uk/user/s20/s2089883/MLP_CW/Coursework3/ml_cw3/Cluster/train_1000_7_fake.png'

# img = Image.open(img_path)
# img.show()
# img = img.resize((32,32))
# img_flat = np.array([])
# img_np = np.asarray(img)
# for i in range(0,3):
# 	img_flat = np.append(img_flat, np.array(img_np[:,:,i].flatten()))
# img_flat = img_flat.astype('uint8')

# #convert back to RGB and show
# red = img_flat[:(32*32)].reshape(32,32)
# green = img_flat[(32*32):(32*32)*2].reshape(32,32)
# blue = img_flat[(32*32)*2:(32*32)*3].reshape(32,32)

# rgb = np.dstack((red,green,blue))
# # i = i.reshape(32,32,-1)
# # print(i.shape)

# img = Image.fromarray(rgb)
# img.show()


trial_file = '/afs/inf.ed.ac.uk/user/s20/s2089883/MLP_CW/Coursework3/ml_cw3/Cluster/MonetBatch1.pickle'
orig_file = '/afs/inf.ed.ac.uk/user/s20/s2089883/MLP_CW/Coursework3/ml_cw3/Cluster/CIFAR10_Data/cifar-10-batches-py/data_batch_1'
read_dict = unpickle(trial_file)
# read_dict = unpickle(orig_file)


i = read_dict['data'][9283]
j = read_dict['labels'][9283]
print(i.shape, j)


# img = Image.open(img_path)
# img_np = np.asarray(img)
# for i in range(0,3):
# 	img_flat = np.append(img_flat, np.array(img_np[:,:,i].flatten()))

# img_flat = img_flat.astype('uint8')

red = i[:(32*32)].reshape(32,32)
green = i[(32*32):(32*32)*2].reshape(32,32)
blue = i[(32*32)*2:(32*32)*3].reshape(32,32)

rgb = np.dstack((red,green,blue))
# i = i.reshape(32,32,-1)
# print(i.shape)

img = Image.fromarray(rgb)
img.show()
