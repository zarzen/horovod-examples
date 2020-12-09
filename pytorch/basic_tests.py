import argparse
import torch.multiprocessing as mp
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd



def test_correctness():
    """"""
    r1 = torch.rand(10, dtype=torch.float32)
    r1 = r1.cuda()
    print('rank', hvd.rank(), 'random array', r1)
    if hvd.size() > 1:
        ret = hvd.allreduce(r1, name="random_tensor1")
        print('allreduced', ret)

if __name__ == '__main__':
    
    hvd.init()
    random_seed = hvd.rank()
    torch.manual_seed(random_seed)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(random_seed)
    test_correctness()