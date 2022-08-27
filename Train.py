"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

import numpy as np
import argparse
from tqdm import tqdm

from models.PointerNet.PointerNet import PointerNet
from models.PointerNet.Data_Generator import TSPDataset

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# TSP
parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')

params = parser.parse_args()

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

dataset = TSPDataset("data/training/TSP", range(4,10), "EUC_2D")
    
model = PointerNet(params.embedding_size, params.batch_size)

dataloader = DataLoader(dataset,
                        num_workers=8,
                        prefetch_factor=4,
                        sampler=torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset), batch_size=params.batch_size, drop_last=True))

if USE_CUDA:
    model.cuda()
    net = model #torch.nn.parallel.DistributedDataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss(reduction='none')
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []




for epoch in range(params.nof_epoch):
    batch_loss = []
    iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Batch %i/%i' % (epoch+1, params.nof_epoch))

        train_batch = sample_batched['data'][0]
        target_batch = sample_batched['tour_ohe'][0]
        sequence_lengths = sample_batched["sequence_length"][0]

        if USE_CUDA:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()
            sequence_lengths = sequence_lengths.cuda()

            
        max_sequence_length = torch.max(sample_batched["sequence_length"]).item()
        o, p = model(train_batch, sample_batched["sequence_length"][0])
        loss = torch.sum(
            torch.nn.utils.rnn.pack_padded_sequence(
                CCE
                (
                    o.permute(1,0,2).reshape(-1,max_sequence_length), 
                    target_batch.reshape(-1,max_sequence_length)
                ).reshape(-1,max_sequence_length) / sequence_lengths.view(-1,1), 
                batch_first=True, 
                lengths=sample_batched["sequence_length"][0].data, 
                enforce_sorted=False
            ).data
        ) / sequence_lengths.shape[0]

        losses.append(loss.item())
        batch_loss.append(loss.item())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        iterator.set_postfix(loss='{}'.format(loss.item()))

    iterator.set_postfix(loss=np.average(batch_loss))
