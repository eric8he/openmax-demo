# standard library imports
import pickle

# third party imports
import libmr
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm, trange

##                                ##
## Stage 1: Evaluation through NN ##
##                                ##

# initialize transforms, model, and data
normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
dataset = datasets.CIFAR10(root="data", train = True, transform = transforms.Compose([transforms.ToTensor(),normalize]), download = True) 
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

# array to store activation vectors (AVs)
avs = [[] for i in range(10)]

# run through all images in dataset and eval
with torch.no_grad():
  for i, data in tqdm(enumerate(dataloader)):
    inputs, labels = data

    # get batch of classifications from model
    outputs = model(inputs)
    
    for output, label in zip(outputs, labels):
      # only add AV to array if example was classified correctly
      if torch.argmax(output) == label:
        # shift and make sure all values are positive
        t = torch.add(output, torch.min(output))

        # add AV to array
        avs[label].append(output.numpy())


##                      ##
## Stage 2: Computation ##
##                      ##

# initialize
means = [0] * 10
models = [None] * 10
dists = [0] * 10

avs_n = [np.array(t) for t in avs]

# compute all means and models
for j in trange(10):
  # initialize new libMR instance (new distribution)
  mr = libmr.MR()

  # compute mean activation vector (MAV) for class
  means[j] = np.mean(avs_n[j], axis=0)

  # compute all distances between MAV and actual AV, store
  s_j_dist = [np.linalg.norm(avs[j][k] - means[j]) for k in range(len(avs[j]))]

  # use libMR to fit Weibull distribution and store
  mr.fit_high(s_j_dist, 20)
  models[j] = mr


##                 ##
## Stage 3: Output ##
##                 ##

# dump precalculated values to a file
arr = [means, models]

with open('array.pickle', 'wb') as handle:
    pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
