from __future__ import print_function, division
import torch

def get_weights(dataset):

    # Get targets
    targets = [dataset.__getitem__(i)[1] for i in range(dataset.__len__())]
    # Count zero value appearance
    count_zeros = len(list(filter(lambda x: x == 0, targets))) 
    # Weights - inverted possibilities
    weights_zeros = float(sum(targets))/float(count_zeros)
    weights_others = float(sum(targets))/float(sum(targets) - count_zeros)
    # Normalized
    weights_zeros = weights_zeros / (weights_zeros + weights_others)  
    weights_others = weights_others / (weights_zeros + weights_others) 
    # Weight for each sample                               
    weights = [ weights_zeros if target == 0 else weights_others for target in targets]

    return weights