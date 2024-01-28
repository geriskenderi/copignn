import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch_geometric.loader import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from data import DRegDataset
from models import COPIGNN
from utils import eval_maxcut, eval_mis

def run(args):
    # Seed
    os.environ["PL_GLOBAL_SEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create datasets and loaders
    in_dim = args.num_nodes ** 0.5 if args.num_nodes >= 1e5 else args.num_nodes ** (1/3)
    in_dim = round(in_dim)
    dataset = DRegDataset(args.node_degree, args.num_graphs, args.num_nodes, in_dim, args.seed)
    print('dataset len:', len(dataset))
    dataloader = DataLoader(dataset.data, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers)
    print('dataloader ready...')

    # Build model
    hidden_dim = round(in_dim/2)
    co_problem = 'maxcut' if args.maxcut else 'mis'
    model = COPIGNN(
        in_dim, 
        hidden_dim, 
        co_problem,
        lr=args.learning_rate,
        out_dim=1, 
        num_heads=args.num_heads, 
        layer_type=args.gnn_model
    )

    # Training (via PyL)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=1000, 
        verbose=True, 
        mode="min"
    )

    trainer = Trainer(
        callbacks=[early_stop_callback],
        devices=[0],
        accelerator='gpu',
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    # Evaluate after training
    model.eval()
    eval_fn = eval_maxcut if co_problem == 'maxcut' else eval_mis

    with torch.no_grad():
        e, a = [], []
        for batch in tqdm(dataloader, desc='evaluating model...'):
            x, edge_index = batch.x, batch.edge_index
            pred = model(x, edge_index)
            proj = torch.round(pred)
            energy, approx_ratio = eval_fn(edge_index, proj, args.node_degree, args.num_graphs)
            e.append(energy.item()), a.append(approx_ratio.item())

    print(f'Avg. estimated energy: {np.mean(e)}, avg. approximation ratio: {np.mean(a)}')    
    print(f'Completed training for seed={args.seed}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_graphs', type=int, default=100)
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--node_degree', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--maxcut', action='store_true', help='If this flag is true solves the maxcut problem, else mis')
    parser.add_argument('--gnn_model', type=int, default=0)
    parser.add_argument('--num_heads', type=int, default=4)
    # parser.add_argument('--logging_info', type=str)

    args = parser.parse_args()
    run(args)


