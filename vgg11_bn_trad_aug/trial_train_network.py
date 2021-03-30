from PIL import Image
import numpy as np
import data_providers as data_providers
# from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from model_architectures import ConvolutionalNetwork

# args, device = get_args()  # get arguments from command line
# rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch

# torch.manual_seed(seed=args.seed)  # sets pytorch's seed

#original transforms
transform_train = transforms.Compose([
	# transforms.RandomCrop(32, padding=4),
	# transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



#trial - no transforms
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

trainset = data_providers.CIFAR10(root='CIFAR10_Data/', set_name='train', download=False, transform=transform_train)
styled = data_providers.AugmentedCIFAR10(path='CIFAR10_Data/Pickled/',transform=transform_train)

concatenated = torch.utils.data.ConcatDataset([trainset, styled])

train_data = torch.utils.data.DataLoader(concatenated, batch_size=128, shuffle=True, num_workers=4)
print('Total elements for training:',len(concatenated),len(styled))

print(styled[9][1].dtype)
print(trainset[9][1].dtype)

# print(styled[0])
# path='CIFAR10_Data/Pickled/'
# import os
# import pickle

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# data = np.array([])
# labels = np.array([])
# pickled_batches = [os.path.join(path,f) for f in os.listdir(path)]
# for pickled_batch in pickled_batches:
# 	entry = unpickle(pickled_batch)
# 	data = np.append(data, entry['data'])
# 	labels = np.append(labels, entry['labels'])


# data = data.reshape((10000, 3, 32, 32))
# data = data.transpose((0, 2, 3, 1))  # convert to HWC
# data = data.astype('uint8')

# print('Augmented',data.shape)
# print('Augmented',labels.shape)

# idx = 9

# img = data[idx]
# print(img.shape)

# img = Image.fromarray(img)
# img.show()







# print(trainset[0])

# valset = data_providers.CIFAR10(root='data', set_name='val', download=False, transform=transform_test)
# val_data = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)

# testset = data_providers.CIFAR10(root='data', set_name='test', download=False, transform=transform_test)
# test_data = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# num_output_classes = 10

# custom_conv_net = ConvolutionalNetwork(num_output_classes=num_output_classes)  # initialize our network object, in this case a ConvNet (VGG-11 with BN)

# conv_experiment = ExperimentBuilder(network_model=custom_conv_net, use_gpu=args.use_gpu,
#                                     experiment_name=args.experiment_name,
#                                     num_epochs=args.num_epochs,
#                                     learning_rate=args.learning_rate,
#                                     weight_decay_coefficient = args.weight_decay,
#                                     continue_from_epoch=args.continue_from_epoch,
#                                     train_data=train_data, val_data=val_data,
#                                     test_data=test_data)  # build an experiment object
# experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics


