"""Custom training script for the RFM (Random Feature Masking) model.

This module implements a customizable training pipeline for deep learning models
using Random Feature Masking (RFM) technique. It provides enhanced functionality
for training on custom datasets with improved error handling, logging, and
configuration options.

The script supports distributed training, model checkpointing, and extensive
hyperparameter configurations through command-line arguments.
"""

import torch
from utils.utils import data_prefetcher_two, cal_fam, setup_seed, calRes
from pretrainedmodels import xception
from utils.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import argparse
import random
import time
import os

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description='Train the model on custom dataset')

# Dataset arguments
parser.add_argument('--dataset_path', type=str, default=r"D:\.THESIS\datasets\sample_data",
                    help='Path to your dataset root directory')

# Model arguments
parser.add_argument('--device', default="cuda:0", type=str,
                    help='Device to use (cuda:0 or cpu)')
parser.add_argument('--modelname', default="xception", type=str,
                    help='Model name')
parser.add_argument('--distributed', default=False, action='store_true',
                    help='Use distributed training')

# Training arguments
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--max_batch', default=100, type=int,
                    help='Maximum number of batches to train')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers for data loading')
parser.add_argument('--logbatch', default=5, type=int,
                    help='Log every N batches')
parser.add_argument('--savebatch', default=5, type=int,
                    help='Save model every N batches')
parser.add_argument('--seed', default=5, type=int,
                    help='Random seed')

# Learning rate
parser.add_argument('--lr', default=0.0005, type=float,
                    help='Learning rate')

# RFM parameters
parser.add_argument('--eH', default=120, type=int,
                    help='Maximum height for RFM masking')
parser.add_argument('--eW', default=120, type=int,
                    help='Maximum width for RFM masking')

# Model checkpointing
parser.add_argument('--pin_memory', '-p', default=False, action='store_true',
                    help='Use pin memory for data loading')
parser.add_argument('--resume_model', default=None,
                    help='Path to resume model from')
parser.add_argument('--resume_optim', default=None,
                    help='Path to resume optimizer from')
parser.add_argument('--save_model', default=True, action='store_true',
                    help='Save model checkpoints')
parser.add_argument('--save_optim', default=False, action='store_true',
                    help='Save optimizer checkpoints')

# File naming
parser.add_argument('--upper', default="xbase", type=str,
                    help='Prefix for saved files')

args = parser.parse_args()
upper = args.upper
modelname = args.modelname

def Eval(model, lossfunc, dtloader):
    """Evaluate the model on a given dataset.

    This function evaluates the model's performance on a dataset, calculating
    loss and predictions for all samples.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        lossfunc (torch.nn.Module): The loss function to use for evaluation.
        dtloader (torch.utils.data.DataLoader): DataLoader containing the evaluation dataset.

    Returns:
        tuple: A tuple containing:
            - float: Average loss over the dataset
            - torch.Tensor: Ground truth labels
            - torch.Tensor: Model predictions
    """
    model.eval()
    sumloss = 0.
    y_true_all = None
    y_pred_all = None

    with torch.no_grad():
        for (j, batch) in enumerate(dtloader):
            x, y_true = batch
            y_pred = model.forward(x.cuda())

            loss = lossfunc(y_pred, y_true.cuda())
            sumloss += loss.detach()*len(x)

            y_pred = torch.nn.functional.softmax(
                y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))

    return sumloss/len(y_true_all), y_true_all.detach(), y_pred_all.detach()

def Log(log):
    """Write log message to both console and log file.

    This function writes a log message to both the console and a log file,
    creating the log directory if it doesn't exist.

    Args:
        log (str): The log message to write.
    """
    print(log)
    os.makedirs("./logs", exist_ok=True)
    f = open("./logs/"+upper+"_"+modelname+".log", "a")
    f.write(log+"\n")
    f.close()

if __name__ == "__main__":
    # Print configuration
    print("\nTraining Configuration:")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Model: {args.modelname}")
    print("----------------------------------------")

    Log("\nModel:%s BatchSize:%d lr:%f" % (modelname, args.batch_size, args.lr))
    torch.cuda.set_device(args.device)
    setup_seed(args.seed)
    print("cudnn.version:%s enabled:%s benchmark:%s deterministic:%s" % (torch.backends.cudnn.version(), torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))

    MAX_TPR_4 = 0.

    # Initialize model
    model = xception(num_classes=2, pretrained=False).cuda()

    if args.distributed:
        model = torch.nn.DataParallel(model)

    optim = Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if args.resume_model is not None:
        print(f"Loading model from {args.resume_model}")
        model.load_state_dict(torch.load(args.resume_model))
    if args.resume_optim is not None:
        print(f"Loading optimizer from {args.resume_optim}")
        optim.load_state_dict(torch.load(args.resume_optim))

    lossfunc = torch.nn.CrossEntropyLoss()

    # Initialize dataset
    print(f"\nLoading dataset from {args.dataset_path}")
    dataset = CustomDataset(folder_path=args.dataset_path)

    # Load datasets
    trainsetR = dataset.getTrainsetR()
    trainsetF = dataset.getTrainsetF()
    validset = dataset.getValidset()
    testsetR = dataset.getTestsetR()
    TestsetList, TestsetName = dataset.getsetlist(real=False, setType=2)

    print(f"Training set - Real: {len(trainsetR)} images, Fake: {len(trainsetF)} images")
    print(f"Validation set: {len(validset)} images")
    print(f"Test set - Real: {len(testsetR)} images, Fake sets: {len(TestsetList)}")

    setup_seed(args.seed)

    # Create data loaders
    traindataloaderR = DataLoader(
        trainsetR,
        batch_size=int(args.batch_size/2),
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    traindataloaderF = DataLoader(
        trainsetF,
        batch_size=int(args.batch_size/2),
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    validdataloader = DataLoader(
        validset,
        batch_size=args.batch_size*2,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    testdataloaderR = DataLoader(
        testsetR,
        batch_size=args.batch_size*2,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    testdataloaderList = []
    for tmptestset in TestsetList:
        testdataloaderList.append(
            DataLoader(
                tmptestset,
                batch_size=args.batch_size*2,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers
            )
        )

    print("\nStarting training...")
    print("----------------------------------------")

    batchind = 0
    e = 0
    sumcnt = 0
    sumloss = 0.
    while True:
        prefetcher = data_prefetcher_two(traindataloaderR, traindataloaderF)

        data, y_true = prefetcher.next()

        while data is not None and batchind < args.max_batch:
            stime = time.time()
            sumcnt += len(data)

            ''' ↓ the implementation of RFM ↓ '''
            model.eval()
            mask = cal_fam(model, data)
            imgmask = torch.ones_like(mask)
            imgh = imgw = 224

            for i in range(len(mask)):
                maxind = np.argsort(mask[i].cpu().numpy().flatten())[::-1]
                pointcnt = 0
                for pointind in maxind:
                    pointx = pointind//imgw
                    pointy = pointind % imgw

                    # Add bounds checking
                    if pointx >= imgh or pointy >= imgw:
                        continue

                    if imgmask[i][0][pointx][pointy] == 1:
                        maskh = random.randint(1, args.eH)
                        maskw = random.randint(1, args.eW)

                        sh = random.randint(1, maskh)
                        sw = random.randint(1, maskw)

                        top = max(pointx-sh, 0)
                        bot = min(pointx+(maskh-sh), imgh)
                        lef = max(pointy-sw, 0)
                        rig = min(pointy+(maskw-sw), imgw)

                        imgmask[i][:, top:bot, lef:rig] = torch.zeros_like(imgmask[i][:, top:bot, lef:rig])

                        pointcnt += 1
                        if pointcnt >= 3:
                            break

            data = imgmask * data + (1-imgmask) * (torch.rand_like(data)*2-1.)
            ''' ↑ the implementation of RFM ↑ '''

            model.train()
            y_pred = model.forward(data)
            loss = lossfunc(y_pred, y_true)

            flood = (loss-0.04).abs() + 0.04
            sumloss += loss.detach()*len(data)
            data, y_true = prefetcher.next()

            optim.zero_grad()
            flood.backward()
            optim.step()

            batchind += 1
            print("Train %06d loss:%.5f avgloss:%.5f lr:%.6f time:%.4f" % (batchind, loss, sumloss/sumcnt, optim.param_groups[0]["lr"], time.time()-stime), end="\r")

            if batchind % args.logbatch == 0:
                print()
                Log("epoch:%03d batch:%06d loss:%.5f avgloss:%.5f" % (e, batchind, loss, sumloss/sumcnt))

                loss_valid, y_true_valid, y_pred_valid = Eval(model, lossfunc, validdataloader)
                ap, acc, AUC, TPR_2, TPR_3, TPR_4 = calRes(y_true_valid, y_pred_valid)
                Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f %s" % (AUC, TPR_2, TPR_3, TPR_4, "validset"))

                loss_r, y_true_r, y_pred_r = Eval(model, lossfunc, testdataloaderR)
                sumAUC = sumTPR_2 = sumTPR_3 = sumTPR_4 = 0
                for i, tmptestdataloader in enumerate(testdataloaderList):
                    loss_f, y_true_f, y_pred_f = Eval(model, lossfunc, tmptestdataloader)
                    ap, acc, AUC, TPR_2, TPR_3, TPR_4 = calRes(torch.cat((y_true_r, y_true_f)), torch.cat((y_pred_r, y_pred_f)))
                    sumAUC += AUC
                    sumTPR_2 += TPR_2
                    sumTPR_3 += TPR_3
                    sumTPR_4 += TPR_4
                    Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f %s" % (AUC, TPR_2, TPR_3, TPR_4, TestsetName[i]))
                if len(testdataloaderList) > 1:
                    Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f Test" %
                        (sumAUC/len(testdataloaderList), sumTPR_2/len(testdataloaderList), sumTPR_3/len(testdataloaderList), sumTPR_4/len(testdataloaderList)))
                    TPR_4 = (sumTPR_4)/len(testdataloaderList)

                if batchind % args.savebatch == 0 or TPR_4 > MAX_TPR_4:
                    MAX_TPR_4 = TPR_4
                    os.makedirs("./models", exist_ok=True)
                    if args.save_model:
                        torch.save(model.state_dict(), "./models/" + upper+"_"+modelname+"_model_batch_"+str(batchind))
                    if args.save_optim:
                        torch.save(optim.state_dict(), "./models/" + upper+"_"+modelname+"_optim_batch_"+str(batchind))

                print("-------------------------------------------")
        e += 1 