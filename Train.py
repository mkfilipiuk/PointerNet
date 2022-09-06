"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import os
from collections import defaultdict

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import numpy as np
import argparse
from tqdm import tqdm
import apex
import neptune.new as neptune

from models.PointerNet.PointerNet import PointerNet
from models.PointerNet.Data_Generator import TSPDataset
from src.hyperparameters import r

USE_CUDA = True

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
    loss = torch.mean(results["loss"])
    p = measure_tour_length(results["point_coordinates"], results['predicted_tour'])
    gt = measure_tour_length(results["point_coordinates"], results['ground_truth'])
    mean_gt = torch.mean(gt)
    mean_p = torch.mean(p)
    gap = torch.mean(p/gt)
    print(f"Tours of length {l}:")
    print(f"Loss: {loss}")
    print(f"Average optimal length: {mean_gt:.5f}\t\tAverage predicted length: {mean_p:.5f}")
    print(f"Average gap:\t\t{gap:.5f}\t\tGap of averages:          {mean_p/mean_gt:.5f}")
    return {
        "loss": loss,
        "average_optimal_length": mean_gt,
        "average_predicted_length": mean_p,
        "average_gap": gap,
        "gap_of_averages": mean_p/mean_gt
    }

def evaluate_model(model, dataset, batch_size, run, mode="validation"):
    model.eval()
    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=2,
        sampler=torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
    )

    total_results = {}

    results = {
        "point_coordinates": [],
        "predicted_tour": [],
        "loss": [],
        "ground_truth": [],
    }

    CCE = torch.nn.CrossEntropyLoss(reduction='none')
    
    current_length = -1

    for i_batch, sample_batched in enumerate(dataloader):
        train_batch = sample_batched['data'][0]
        target_batch = sample_batched['tour_ohe'][0]
        sequence_lengths = sample_batched["sequence_length"][0]
        if current_length == -1:
            current_length = sequence_lengths.view(-1)[0].item()
        elif current_length != sequence_lengths.view(-1)[0].item():
            for c in ["point_coordinates", "predicted_tour", "ground_truth", "loss"]:
                results[c] = torch.cat(results[c])
            total_results[current_length] = analyse_results(current_length, results)

            results = {
                "point_coordinates": [],
                "predicted_tour": [],
                "loss": [],
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

        o, p = model(train_batch, sample_batched["sequence_length"][0], max_sequence_length)
        o = o.detach().cpu()
        p = p.detach().cpu()
        
        loss = torch.sum(
            torch.nn.utils.rnn.pack_padded_sequence(
                CCE
                (
                    o.permute(1,0,2).reshape(-1,max_sequence_length), 
                    target_batch.cpu().reshape(-1,max_sequence_length)
                ).reshape(-1,max_sequence_length) / sequence_lengths.cpu().view(-1,1), 
                batch_first=True, 
                lengths=sample_batched["sequence_length"][0], 
                enforce_sorted=False
            ).data
        ) / sequence_lengths.shape[0]

        results["predicted_tour"].append(p)
        results["loss"].append(loss.reshape(-1))

    for c in ["point_coordinates", "predicted_tour", "ground_truth", "loss"]:
        results[c] = torch.cat(results[c])
    total_results[current_length] = analyse_results(current_length, results)
    model.train()
    
    average_optimal_length_list = []
    average_predicted_length_list = []
    average_gap_list = []
    gap_of_averages_list = []
    loss_list = []
    for k, v in total_results.items():
        for k2, v2 in v.items():
            run[f"results_{mode}/{k}/{k2}"].log(v2)
            eval(f"{k2}_list").append(v2)
            
    run[f"results_{mode}/averaged/loss"].log(sum(loss_list)/len(loss_list))
    run[f"results_{mode}/averaged/average_optimal_length"].log(sum(average_optimal_length_list)/len(average_optimal_length_list))
    run[f"results_{mode}/averaged/average_predicted_length"].log(sum(average_predicted_length_list)/len(average_predicted_length_list))
    run[f"results_{mode}/averaged/average_gap"].log(sum(average_gap_list)/len(average_gap_list))
    run[f"results_{mode}/averaged/gap_of_averages"].log(sum(gap_of_averages_list)/len(gap_of_averages_list))
    return total_results


def train(gpu, params, run):
    if params.gpu_number > 0:
        cudnn.benchmark = True
    if params.gpu_number > 1 and params.DDP:
        torch.distributed.init_process_group(backend='nccl', world_size=params.gpu_number, rank=gpu)
        torch.cuda.set_device(gpu)
        torch.cuda.empty_cache()
    
    if params.data_source == "ptrnet_data":
        training_dataset = TSPDataset("data/ptrnet_data/train", range(params.train_low,params.train_high), params.metric, params.train_size, data_source="ptrnet_data", omit_tour_length=True, data_in_ints=False)
        validation_dataset = TSPDataset("data/ptrnet_data/test", range(params.train_low,params.train_high), params.metric, params.validation_size, data_source="ptrnet_data", omit_tour_length=True, data_in_ints=False)
    elif params.data_source == "uniform":
        training_dataset = TSPDataset("data/training/TSP", range(params.train_low,params.train_high), params.metric, params.train_size)
        validation_dataset = TSPDataset("data/validation/TSP", range(4,101), params.metric, params.validation_size)
    test_dataset = TSPDataset("data/test/TSP", [x for x in r if x <= 100], params.metric, params.test_size)
    very_long_test_dataset = TSPDataset("data/test/TSP", [x for x in r if x > 100], params.metric, params.very_long_test_size)

    model = PointerNet(params.embedding_size, params.hidden)
    if params.gpu_number > 0:
        model = model.cuda()
        
    if params.gpu_number > 1 and params.DDP:
        model = DDP(model)
        sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, num_replicas=params.gpu_number, rank=gpu)
    elif params.gpu_number > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(params.gpu_number)))
        sampler = torch.utils.data.RandomSampler(training_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(training_dataset)

    training_dataloader = DataLoader(
        training_dataset,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=4,
        sampler=torch.utils.data.BatchSampler(sampler, batch_size=params.batch_size, drop_last=True)
    )
        

    CCE = torch.nn.CrossEntropyLoss(reduction='none')
    model_optim = apex.optimizers.FusedAdam(
        model.parameters(),
        lr=params.lr,
        weight_decay=0.001
    )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, factor=0.5, patience=10, verbose=True)
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


            max_sequence_length = torch.max(sequence_lengths).item()
            if params.AMP:
                with torch.autocast("cuda"):
                    o, _ = model(train_batch, sample_batched["sequence_length"][0], max_sequence_length)
                    loss = torch.sum(
                        torch.nn.utils.rnn.pack_padded_sequence(
                            CCE
                            (
                                o.permute(1,0,2).reshape(-1,max_sequence_length), 
                                target_batch.reshape(-1,max_sequence_length)
                            ).reshape(-1,max_sequence_length) / sequence_lengths.view(-1,1), 
                            batch_first=True, 
                            lengths=sample_batched["sequence_length"][0], 
                            enforce_sorted=False
                        ).data
                    ) / sequence_lengths.shape[0]
            else:
                o, _ = model(train_batch, sample_batched["sequence_length"][0], max_sequence_length)
                loss = torch.sum(
                    torch.nn.utils.rnn.pack_padded_sequence( 
                        CCE
                        (
                            o.permute(1,0,2).reshape(-1,max_sequence_length), 
                            target_batch.reshape(-1,max_sequence_length)
                        ).reshape(-1,max_sequence_length) / sequence_lengths.view(-1,1), 
                        batch_first=True, 
                        lengths=sample_batched["sequence_length"][0], 
                        enforce_sorted=False
                    ).data
                ) / sequence_lengths.shape[0]


            if params.AMP:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            model_optim.zero_grad()

            losses.append(loss.item())
            batch_loss.append(loss.item())



            iterator.set_postfix(loss='{}'.format(loss.item()))

        summary = evaluate_model(model, validation_dataset, params.validation_batch_size, run)          

        iterator.set_postfix(loss=np.average(batch_loss))
        # scheduler.step(sum([summary[x]["average_gap"] for x in summary]))

    evaluate_model(model, test_dataset, params.validation_batch_size, run, mode="test")
    evaluate_model(model, very_long_test_dataset, params.very_long_test_batch_size, run, mode="test")
    
    with open("checkpoints/ptrnet_model_arch.txt", "w") as f: f.write(str(model))
    torch.save(model, "checkpoints/ptrnet_model.pt")
    run["model_checkpoints/model_arch"].upload("checkpoints/ptrnet_model_arch.txt")
    run["model_checkpoints/model"].upload("checkpoints/ptrnet_model.pt")

def main():
    parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

    # Data
    parser.add_argument('--data_source', default="uniform", type=str, help='Data source')
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
    parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # GPU
    parser.add_argument('--gpu_number', default=1, type=int, help='Number of GPUs for training')
    parser.add_argument('--AMP', default=False, action='store_true', help='Use mixed precision training')
    parser.add_argument('--DDP', default=False, action='store_true', help='Use PyTorch DistributedDataParallel')
    # TSP
    parser.add_argument('--metric', type=str, default="EUC_2D", help='Metric used in TSP')
    # Network
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
    parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units')

    parser.add_argument('--tags', type=str, default="", help='Number of hidden units')
    
    params = parser.parse_args()

    print(params)
    run = neptune.init(
        # model="RLIN-PTRNET",
        project="TensorCell/RLinVRP",
        api_token = os.environ['NEPTUNE_API_TOKEN'],
        source_files = ['**/*.py']
    )
    run["parameters"] = vars(params)
    run["sys/tags"].add(params.tags.split(","))

    if params.gpu_number > 0 and torch.cuda.is_available():
        print('Using GPU, %i devices.' % params.gpu_number)
        
    if params.gpu_number > 1 and params.DDP:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'[:(params.gpu_number*2)-1]
        os.environ['NCCL_DEBUG'] = 'INFO'
        mp.set_start_method("spawn", force=True)
        mp.spawn(train, nprocs=params.gpu_number, args=(params,))
    else:
        train(0, params, run)
        
    run.stop()
    
if __name__ == '__main__':
    main()