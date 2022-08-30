"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

from collections import defaultdict

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from models.PointerNet.PointerNet import PointerNet
from models.PointerNet.Data_Generator import TSPDataset
from src.hyperparameters import r

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--train_low', default=4, type=int, help='Training data size')
parser.add_argument('--train_high', default=101, type=int, help='Training data size')
parser.add_argument('--train_size', default=100000, type=int, help='Test data size')
parser.add_argument('--validation_size', default=10000, type=int, help='Test data size')
parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
parser.add_argument('--very_long_test_size', default=1000, type=int, help='Test data size')
parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
parser.add_argument('--validation_batch_size', default=200, type=int, help='Batch size')
parser.add_argument('--very_long_test_batch_size', default=1, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
parser.add_argument('--AMP', default=True, action='store_true', help='Use mixed precision training')
# TSP
parser.add_argument('--metric', type=str, default="EUC_2D", help='Metric used in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=False, action='store_true', help='Bidirectional')

params = parser.parse_args()

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

training_dataset = TSPDataset("data/training/TSP", range(params.train_low,params.train_high), params.metric, params.train_size)
validation_dataset = TSPDataset("data/validation/TSP", range(params.train_low,params.train_high), params.metric, params.validation_size)
test_dataset = TSPDataset("data/test/TSP", [x for x in r if x <= 100], params.metric, params.test_size)
very_long_test_dataset = TSPDataset("data/test/TSP", [x for x in r if x > 100], params.metric, params.very_long_test_size)
    
model = PointerNet(params.embedding_size, params.hidden, params.nof_lstms)

training_dataloader = DataLoader(
    training_dataset,
    pin_memory=True,
    num_workers=8,
    prefetch_factor=4,
    sampler=torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(training_dataset), batch_size=params.batch_size, drop_last=True)
)

def measure_tour_length(coordinates, tours):
    return torch.sum(
        torch.norm(
            coordinates[
                torch.arange(coordinates.shape[0]).unsqueeze(1), 
                torch.roll(tours, shifts=(0, 1), dims=(0, 1))
            ] - \
            coordinates[
                torch.arange(coordinates.shape[0]).unsqueeze(1), 
                tours
            ], 
            dim=2
        ),
        dim=1
    )


def analyse_results(l, results):
    p = measure_tour_length(results["point_coordinates"], results['predicted_tour'])
    gt = measure_tour_length(results["point_coordinates"], results['ground_truth'])
    mean_gt = torch.mean(gt)
    mean_p = torch.mean(p)
    gap = torch.mean(p/gt)
    print(f"Tours of length {l}:")
    print(f"Average optimal length: {mean_gt:.5f}\t\tAverage predicted length: {mean_p:.5f}")
    print(f"Average gap:\t\t{gap:.5f}\t\tGap of averages:          {mean_p/mean_gt:.5f}")

def evaluate_model(model, dataset, batch_size):
    model.eval()
    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=2,
        sampler=torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
    )
    
    results = {
        "point_coordinates": [],
        "predicted_tour": [],
        "ground_truth": [],
    }
    
    current_length = -1
    
    for i_batch, sample_batched in enumerate(dataloader):
        train_batch = sample_batched['data'][0]
        target_batch = sample_batched['tour_ohe'][0]
        sequence_lengths = sample_batched["sequence_length"][0]
        if current_length == -1:
            current_length = sequence_lengths.view(-1)[0].item()
        elif current_length != sequence_lengths.view(-1)[0].item():
            for c in ["point_coordinates", "predicted_tour", "ground_truth"]:
                results[c] = torch.cat(results[c])
            analyse_results(current_length, results)
            
            results = {
                "point_coordinates": [],
                "predicted_tour": [],
                "ground_truth": [],
            }
            current_length = sequence_lengths.view(-1)[0].item()
        
        results["point_coordinates"].append(sample_batched["data"][0].detach().cpu())
        results["ground_truth"].append(sample_batched["tour"][0].detach().cpu())
        
        if USE_CUDA:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()
            sequence_lengths = sequence_lengths.cuda()

        max_sequence_length = torch.max(sample_batched["sequence_length"]).item()
        
        _, p = model(train_batch, sample_batched["sequence_length"][0])
        p = p.detach().cpu()
        
        l = sequence_lengths.view(-1)[0].item()
        
        results["predicted_tour"].append(p)
        
    for c in ["point_coordinates", "predicted_tour", "ground_truth"]:
        results[c] = torch.cat(results[c])
    analyse_results(current_length, results)
    model.train()


if USE_CUDA:
    model.cuda()
    net = model #torch.nn.parallel.DistributedDataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss(reduction='none')
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []

if params.AMP:
    scaler = torch.cuda.amp.GradScaler()


for epoch in range(params.nof_epoch):
    batch_loss = []
    iterator = tqdm(training_dataloader, unit='Batch')
    model_optim.zero_grad()
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
        if params.AMP:
            with torch.autocast("cuda"):
                o, _ = model(train_batch, sample_batched["sequence_length"][0])
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
        else:
            o, _ = model(train_batch, sample_batched["sequence_length"][0])
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
        
            
        if params.AMP:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()
            
        model_optim.zero_grad()

        losses.append(loss.item())
        batch_loss.append(loss.item())

        

        iterator.set_postfix(loss='{}'.format(loss.item()))
        
    evaluate_model(model, validation_dataset, params.validation_batch_size)          

    iterator.set_postfix(loss=np.average(batch_loss))

evaluate_model(model, test_dataset, params.validation_batch_size)
evaluate_model(model, very_long_test_dataset, params.very_long_test_batch_size)
save_results(results)