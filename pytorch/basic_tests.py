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
    # broadcast_tensor = torch.rand(10, dtype=torch.float32)
    # hvd.broadcast(broadcast_tensor, 0)
    # print("="* 10, 'after broadcast')

    print("-" * 10, 'test 10 random vals')
    r1 = torch.rand(10, dtype=torch.float32)
    r1 = r1.cuda()
    print('rank', hvd.rank(), 'random array', r1)
    ret = hvd.allreduce(r1, name="random_tensor1")
    print('allreduced', ret)

    print('-'* 10, 'test 1 random val in-place')
    one = torch.rand(1, dtype=torch.float32)
    one = one.cuda()
    print('rank', hvd.rank(), 'rand one', one)
    one = hvd.allreduce_(one, name="allreduce_one_item")
    print('allreduce', one)

    print('-' * 10, 'test 1 random val out-place')
    one = torch.rand(1, dtype=torch.float32)
    one = one.cuda()
    print('rank', hvd.rank(), 'rand one', one)
    one = hvd.allreduce(one, name="allreduce_one_item")
    print('allreduce', one)


if __name__ == '__main__':
    
    hvd.init(model_bw_order_file='torch-mnist')
    random_seed = hvd.rank()
    torch.manual_seed(random_seed)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(random_seed)
    test_correctness()