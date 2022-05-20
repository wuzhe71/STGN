import argparse, random, time, os, pdb
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

import np_transforms as NP_T
from CrowdDataset import TestSeq
from model import STGN
from sklearn.metrics import mean_squared_error,mean_absolute_error

def main():
    parser = argparse.ArgumentParser(
        description='Train CSRNet in Crowd dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='STGN.pth', type=str)
    parser.add_argument('--dataset', default='Mall', type=str)
    parser.add_argument('--valid', default=0, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gamma', default=5, type=float)
    parser.add_argument('--max_len', default=4, type=int)
    parser.add_argument('--channel', default=128, type=int)
    parser.add_argument('--block_num', default=4, type=int)
    parser.add_argument('--shape', default=[360, 480], nargs='+', type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_all', action='store_true', help='')
    parser.add_argument('--adaptive', action='store_true', help='')
    parser.add_argument('--agg', action='store_true', help='')
    parser.add_argument('--use_cuda', default=True, type=bool)

    args = vars(parser.parse_args())
    
    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'
    print('device:', device)

    valid_transf = NP_T.ToTensor() 

    datasets = ['TRANCOS', 'Venice', 'UCSD', 'Mall', 'FDST']
    for dataset in datasets:
        if dataset == 'UCSD':
            args['shape'] = [360, 480]
            args['max_len'] = 10
            args['channel'] = 128
        elif dataset == 'Mall':
            args['shape'] = [480, 640]
            args['max_len'] = 4
            args['channel'] = 128
        elif dataset == 'FDST':
            args['max_len'] = 4
            args['shape'] = [360, 640]
            args['channel'] = 128
        elif dataset == 'Venice':
            args['max_len'] = 8
            args['shape'] = [360, 640]
            args['channel'] = 128
        elif dataset == 'TRANCOS':
            args['max_len'] = 4
            args['shape'] = [360, 480]
            args['channel'] = 128
            
        dataset_path = os.path.join('E://code//Traffic//Counting//Datasets', dataset)
        valid_data = TestSeq(train=False,
                             path=dataset_path,
                             out_shape=args['shape'],
                             transform=valid_transf,
                             gamma=args['gamma'],
                             max_len=args['max_len'], 
                             load_all=args['load_all'])
        valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1)

        model = STGN(args).to(device)
        model.eval()
        model_name = os.path.join('./models', dataset, 'STGN.pth')
        assert os.path.exists(model_name) is True
        model.load_state_dict(torch.load(model_name))
        print('Load pre-trained model')

        X, density, count = None, None, None
        
        preds = {}
        predictions = []
        counts = []
        for i, (X, count, seq_len, names) in enumerate(valid_loader):
            X, count, seq_len = X.to(device), count.to(device), seq_len.to(device)

            with torch.no_grad():
                density_pred, count_pred = model(X)
        
            N = torch.sum(seq_len)
            count = count.sum(dim=[2,3,4])
            count_pred = count_pred.data.cpu().numpy()
            count = count.data.cpu().numpy()

            for i, name in enumerate(names):
                dir_name, img_name = name[0].split('&')
                preds[dir_name + '_' + img_name] = count[0, i]
                
                predictions.append(count_pred[0, i])
                counts.append(count[0, i])
                
        mae = mean_absolute_error(predictions, counts)
        rmse = np.sqrt(mean_squared_error(predictions, counts))
        
        print('Dataset : {} MAE : {:.3f} MSE : {:.3f}'.format(dataset, mae, rmse))

        
if __name__ == '__main__':
    main()
