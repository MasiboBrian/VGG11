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

style='Monet'
batch = 'Batch2'
images = [os.path.join('./'+batch+'/',f) for f in os.listdir('./'+batch+'/')]
img_flat = np.array([])
labels = np.array([])

for img_path in images:
	img = Image.open(img_path)
	img = img.resize((32,32)) #need to resize to bring te 256x256 image down to the size of original CIFAR images
	img_np = np.asarray(img)
	for i in range(0,3):
		img_flat = np.append(img_flat, np.array(img_np[:,:,i].flatten()))

	img_flat = img_flat.astype('uint8')

	filename = (img_path.split('/'))[-1]
	label = filename.split('_')[2]
	labels = np.append(labels, label)
	labels = labels.astype('uint8')

final_data = img_flat.reshape(-1,32*32*3)

aug_dict = {'data':final_data, 'labels':labels}

to_pickle(aug_dict, style+batch+'.pickle')
