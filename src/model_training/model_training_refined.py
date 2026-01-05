import torch

import sys
import os
import pandas as pd
import numpy as np
import pickle
import anndata as ad
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.utils.data import Dataset, Subset, DataLoader, Sampler
from collections import defaultdict
import torch.nn.functional as F
from sklearn.model_selection import KFold

from catalyst.utils import set_global_seed

import MLP

from processing_image_features import process_image_features
from sklearn.model_selection import train_test_split


class HierarchicalDataset(Dataset):
    def __init__(self, x_scaled, n_cells, coords, y):
        self.samples = []
        self.coarse_totals = defaultdict(float)
        self.coarse_groups = defaultdict(list)

        for spot in np.unique(coords):
            n_coords = np.sum(coords == spot)
            select_y = np.array(y.iloc[y.index.get_loc(spot), :])
            for fine_spot in range(n_coords):
                x = x_scaled[coords == spot, :][fine_spot]
                num_cells = n_cells[coords == spot][fine_spot]
                idx = len(self.samples)
                self.samples.append({
                    "x": torch.tensor(x).to(torch.float32),
                    # "y": y[spot],
                    "y": select_y,
                    "cells": num_cells,
                    "coarse_id": spot,
                    "idx": idx
                })
                self.coarse_groups[spot].append(idx)
            self.coarse_totals[spot] = select_y

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def get_total_for_coarse_id(self, coarse_id):
        return self.coarse_totals[coarse_id]

    def get_indices_by_coarse(self):
        return list(self.coarse_groups.values())


# ---- Custom sampler to yield all samples per coarse group ----
class CoarseGroupSampler(Sampler):
    def __init__(self, dataset, group_ids):
        self.dataset = dataset
        self.batches = []
        for gid in group_ids:
            self.batches.append(dataset.coarse_groups[gid])

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


# ---- Collate function ----
def collate_fn(batch):
    xs = torch.stack([item["x"] for item in batch])
    ys = torch.tensor([item["y"] for item in batch]).to(torch.float32)
    cells = torch.tensor([item["cells"] for item in batch]).to(torch.float32)
    group_id = batch[0]["coarse_id"]
    return xs, ys, cells, group_id


def gene_loss(preds, num_cells, group_id, dataset):
    gene_counts = preds * torch.tensor(num_cells).unsqueeze(1)
    total_counts = torch.sum(gene_counts, axis=0)
    target_sum = torch.tensor(dataset.get_total_for_coarse_id(group_id)).to(torch.float32)
    output = F.mse_loss(total_counts, target_sum)

    return output


def training(adata, pt_model_dir: str, radius, refine_factor, feature_pd_dir, trained_model_dir, save_model_dir,
             hidden_dims=[2048, 2048, 1024, 1024],
             activations=None, residual=False, norm=False, out_act= nn.ReLU(), include_tanhshrink=False, sample_id=None, training_mode=None, kfold=5):
    codex_marker_array, counts_marker_array, other_marker_array = process_image_features(adata, pt_model_dir, radius,
                                                                                         refine_factor)
    adata_raw = adata
    adata_raw.layers['counts'] = adata_raw.X.copy()
    adata_raw.var_names_make_unique()

    with open(feature_pd_dir, 'rb') as f:
        dbit_pd = pickle.load(f)
    f.close()

    barcodes = np.unique(other_marker_array['Barcode'])
    dbit_pd = dbit_pd.loc[barcodes, :]

    selected_refined_spots = counts_marker_array[:, 0] != 0
    selected_barcodes = np.unique(np.array(other_marker_array['Barcode'])[selected_refined_spots])
    dbit_pd_sub = dbit_pd.loc[selected_barcodes,]

    gene_exp = np.array(dbit_pd_sub)

    counts_marker_array_sub = counts_marker_array[selected_refined_spots, :]
    codex_marker_array_sub = codex_marker_array[selected_refined_spots, :]
    other_marker_array_sub = other_marker_array.iloc[selected_refined_spots, :]

    cellvit_marker_array = np.hstack((codex_marker_array_sub, counts_marker_array_sub))

    if training_mode is None:
        ind_x_train = range(gene_exp.shape[0])
        y_train = gene_exp
    elif training_mode == 'Full':
        ind_x_train, ind_x_test, y_train, y_test = train_test_split(range(gene_exp.shape[0]), gene_exp, test_size=0.2,
                                                                    random_state=123)
    elif training_mode == 'Half':
        dbit_row_index = [int(x.split('-')[0]) for x in dbit_pd_sub.index]
        dbit_col_index = [int(x.split('-')[1]) for x in dbit_pd_sub.index]
        selected_train = [x % 2 == 1 for x in dbit_col_index]
        selected_test = [x % 2 == 0 for x in dbit_col_index]

        ind_x_train = np.where(selected_train)[0]
        ind_x_test = np.where(selected_test)[0]
        y_train = gene_exp[selected_train, :]
        y_test = gene_exp[selected_test, :]
    elif training_mode == 'Quarter':
        dbit_row_index = [int(x.split('-')[0]) for x in dbit_pd_sub.index]
        dbit_col_index = [int(x.split('-')[1]) for x in dbit_pd_sub.index]
        selected_train = [(x % 2 == 1) * (y % 2 == 1) for (x, y) in zip(dbit_col_index, dbit_row_index)]
        selected_test = [(x % 2 == 0) + (y % 2 == 0) for (x, y) in zip(dbit_col_index, dbit_row_index)]
        ind_x_train = np.where(selected_train)[0]
        ind_x_test = np.where(selected_test)[0]
        y_train = gene_exp[selected_train, :]
        y_test = gene_exp[selected_test, :]

    barcodes_train_unique = np.array(dbit_pd_sub.index[ind_x_train])
    ind_train_coord = [i for i in range(other_marker_array_sub.shape[0]) if
                       other_marker_array_sub.iloc[i, 0] in barcodes_train_unique]

    print('number of selected original spots in training: ', len(barcodes_train_unique))
    print('number of selected refined spots in training: ', len(ind_train_coord))

    x_train = cellvit_marker_array[ind_train_coord, :]
    other_marker_array_sub_final = other_marker_array_sub.iloc[ind_train_coord, :]
    barcodes_train = np.array(other_marker_array_sub.iloc[:, 0])[ind_train_coord]
    num_cells_train = np.array(counts_marker_array_sub[:, 0])[ind_train_coord]

    CODEX_HIST_scaler = MinMaxScaler(feature_range=(0, 1))
    CODEX_HIST_scaler.fit(x_train)
    x_train_scaled = CODEX_HIST_scaler.transform(x_train)

    ## TODO save scaler here
    with open(f'./{sample_id}_scaler_refine_level{refine_factor}.pickle', 'wb') as file:
        pickle.dump(CODEX_HIST_scaler, file)

    radius /= refine_factor

    train_coordx = other_marker_array_sub.iloc[
                       ind_train_coord, other_marker_array_sub.columns.get_loc('lower_x_refined')] + radius
    train_coordy = other_marker_array_sub.iloc[
                       ind_train_coord, other_marker_array_sub.columns.get_loc('lower_y_refined')] + radius

    with open(f'./{sample_id}_training_data_level{refine_factor}.pickle', 'wb') as file:
        pickle.dump((x_train, y_train, train_coordx, train_coordy, x_train_scaled, other_marker_array_sub_final), file)

    dataset = HierarchicalDataset(x_train_scaled, num_cells_train, barcodes_train, dbit_pd_sub)

    coarse_ids = list(dataset.coarse_groups.keys())

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    ## set up optimizer and loss
    model = MLP.Model(x_train.shape[-1], y_train.shape[-1], hidden_dims, activations, residual, norm, out_act,
                      include_tanhshrink)
    model.load_state_dict(torch.load(
        trained_model_dir,
        map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load('./logs_all_genes_HumanKidney_Visium_HD_64um_counts_normal_hvg1000_2x/best_model_fold5.pth', map_location=torch.device('cpu')))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Set global random seed
    set_global_seed(123)

    best_val_loss = float("inf")

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    for fold, (train_gids, val_gids) in enumerate(kf.split(coarse_ids)):
        print(f"\n Fold {fold + 1}")

        train_sampler = CoarseGroupSampler(dataset, [coarse_ids[i] for i in train_gids])
        val_sampler = CoarseGroupSampler(dataset, [coarse_ids[i] for i in val_gids])

        train_loader = DataLoader(dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
        val_loader = DataLoader(dataset, batch_sampler=val_sampler, collate_fn=collate_fn)
        best_val_loss = float("inf")

        for epoch in range(50):  # keep short for demo
            model.train()
            for x_batch, y_batch, num_cells, group_id in train_loader:
                preds = model(x_batch)
                loss = gene_loss(preds, num_cells, group_id, dataset)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation (optional)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch, num_cells, group_id in val_loader:
                    preds = model(x_batch)
                    val_loss += gene_loss(preds, num_cells, group_id, dataset).item()

            print(f"Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           f"{save_model_dir}/best_model_fold{fold + 1}.pth")
                print(f"Saved best model for fold {fold + 1} at epoch {epoch + 1}")