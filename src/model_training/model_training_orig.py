import torch
from torch import nn

from torch.utils.data import DataLoader, Subset, TensorDataset

import pandas as pd
import numpy as np
import pickle
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler

from src.model_training.runner import CustomRunner
from src.model_training import MLP

from sklearn.model_selection import KFold
from catalyst.utils import set_global_seed

from src.model_training.processing_image_features import process_image_features
from sklearn.model_selection import train_test_split



def training_orig(adata, pt_model_dir: str, radius, feature_list, save_model_dir, hidden_dims=[2048, 2048, 1024, 1024],
                  activations=None, residual=False, norm=False, out_act= nn.ReLU(), include_tanhshrink=False,
                  sample_id=None, training_mode=None, selected_model=None, kfold=5, cell_embed_dir=None, num_epochs=50, sort_way='row_col', datatype='Visium'):
    codex_marker_array, counts_marker_array, other_marker_array = process_image_features(adata, pt_model_dir, radius, 1, cell_embed_dir, sort_way)

    adata_raw = adata
    adata_raw.layers['counts'] = adata_raw.X.copy()
    adata_raw.var_names_make_unique()

    adata_filtered = adata_raw[:, feature_list]
    sc.pp.calculate_qc_metrics(adata_filtered, inplace=True)
    adata_filtered_sub = adata_filtered
    dbit_pd = pd.DataFrame.sparse.from_spmatrix(adata_filtered_sub.layers['counts'], index=adata_filtered_sub.obs_names,
                                                columns=adata_filtered_sub.var_names)

    dbit_pd = dbit_pd.loc[other_marker_array['Barcode'], :]

    ## TODO save dbit_pd
    with open(f'./{sample_id}_gene_exp_pd.pickle', 'wb') as file:
        pickle.dump(dbit_pd, file)

    spots_selected = [i for i in range(counts_marker_array.shape[0]) if counts_marker_array[i, 0] != 0]

    counts_marker_array_sub = counts_marker_array[spots_selected, :]
    codex_marker_array_sub = codex_marker_array[spots_selected, :]
    other_marker_array_sub = other_marker_array.iloc[spots_selected, :]

    if datatype == 'dbit':
        selected_spots = np.array(other_marker_array_sub.Barcode)
        selected_spots_final = [spot for spot in selected_spots if spot in dbit_pd.index]
        selected_spots_final_index = [i for i in range(len(selected_spots)) if selected_spots[i] in dbit_pd.index]
        dbit_pd_sub = dbit_pd.loc[selected_spots_final, :].copy()

        other_marker_array_sub = other_marker_array_sub.iloc[selected_spots_final_index, :]
        codex_marker_array_sub = codex_marker_array_sub[selected_spots_final_index, :]
        counts_marker_array_sub = counts_marker_array_sub[selected_spots_final_index, :]
    else:
        dbit_pd_sub = dbit_pd.iloc[spots_selected, :]

    gene_exp = np.array(dbit_pd_sub) / counts_marker_array_sub[:, 0][:, None]

    cellvit_marker_array = np.hstack((codex_marker_array_sub, counts_marker_array_sub))

    print(cellvit_marker_array.shape)

    if training_mode is None:
        x_train = cellvit_marker_array
        y_train = gene_exp

        train_coordx = np.array(other_marker_array_sub.loc[:, 'pxl_col_in_fullres'])
        train_coordy = np.array(other_marker_array_sub.loc[:, 'pxl_row_in_fullres'])
    elif training_mode == 'Full':
        x_train, x_test, y_train, y_test = train_test_split(cellvit_marker_array, gene_exp, test_size=0.3, random_state=123)
        train_coordx, test_coordx, train_coordy, test_coordy = train_test_split(
            np.array(other_marker_array_sub.loc[:, 'pxl_col_in_fullres']),
            np.array(other_marker_array_sub.loc[:, 'pxl_row_in_fullres']), test_size=0.3, random_state=123)
    elif training_mode == 'Full_2':
        x_train, x_test, y_train, y_test = train_test_split(cellvit_marker_array, gene_exp, test_size=0.2, random_state=123)
        train_coordx, test_coordx, train_coordy, test_coordy = train_test_split(
            np.array(other_marker_array_sub.loc[:, 'pxl_col_in_fullres']),
            np.array(other_marker_array_sub.loc[:, 'pxl_row_in_fullres']), test_size=0.2, random_state=123)
    elif training_mode == 'Full_3':
        x_train, x_test, y_train, y_test = train_test_split(cellvit_marker_array, gene_exp, test_size=0.1, random_state=123)
        train_coordx, test_coordx, train_coordy, test_coordy = train_test_split(
            np.array(other_marker_array_sub.loc[:, 'pxl_col_in_fullres']),
            np.array(other_marker_array_sub.loc[:, 'pxl_row_in_fullres']), test_size=0.1, random_state=123)
    elif training_mode == 'Half':
        x_train = cellvit_marker_array[other_marker_array_sub.array_col % 2 == 1, :]
        x_test = cellvit_marker_array[other_marker_array_sub.array_col % 2 == 0, :]
        y_train = gene_exp[other_marker_array_sub.array_col % 2 == 1, :]
        y_test = gene_exp[other_marker_array_sub.array_col % 2 == 0, :]

        train_coordx = np.array(other_marker_array_sub.loc[:, 'pxl_col_in_fullres'])[
            other_marker_array_sub.array_col % 2 == 1]
        test_coordx = np.array(other_marker_array_sub.loc[:, 'pxl_col_in_fullres'])[
            other_marker_array_sub.array_col % 2 == 0]
        train_coordy = np.array(other_marker_array_sub.loc[:, 'pxl_row_in_fullres'])[
            other_marker_array_sub.array_col % 2 == 1]
        test_coordy = np.array(other_marker_array_sub.loc[:, 'pxl_row_in_fullres'])[
            other_marker_array_sub.array_col % 2 == 0]
    elif training_mode == 'Quarter':
        x_train = cellvit_marker_array[
                  (other_marker_array_sub.array_col % 2 == 1) * (other_marker_array_sub.array_row % 2 == 1), :]
        x_test = cellvit_marker_array[
                 (other_marker_array_sub.array_col % 2 == 0) + (other_marker_array_sub.array_row % 2 == 0), :]
        y_train = gene_exp[(other_marker_array_sub.array_col % 2 == 1) * (other_marker_array_sub.array_row % 2 == 1), :]
        y_test = gene_exp[(other_marker_array_sub.array_col % 2 == 0) + (other_marker_array_sub.array_row % 2 == 0), :]

        train_coordx = np.array(other_marker_array_sub.loc[:, 'pxl_col_in_fullres'])[
            (other_marker_array_sub.array_col % 2 == 1) * (other_marker_array_sub.array_row % 2 == 1)]
        test_coordx = np.array(other_marker_array_sub.loc[:, 'pxl_col_in_fullres'])[
            (other_marker_array_sub.array_col % 2 == 0) + (other_marker_array_sub.array_row % 2 == 0)]
        train_coordy = np.array(other_marker_array_sub.loc[:, 'pxl_row_in_fullres'])[
            (other_marker_array_sub.array_col % 2 == 1) * (other_marker_array_sub.array_row % 2 == 1)]
        test_coordy = np.array(other_marker_array_sub.loc[:, 'pxl_row_in_fullres'])[
            (other_marker_array_sub.array_col % 2 == 0) + (other_marker_array_sub.array_row % 2 == 0)]



    CODEX_HIST_scaler = MinMaxScaler(feature_range=(0, 1))
    CODEX_HIST_scaler.fit(x_train)
    x_train_scaled = CODEX_HIST_scaler.transform(x_train)

    with open(f'./{sample_id}_training_data_level1.pickle', 'wb') as file:
        pickle.dump((x_train, y_train, train_coordx, train_coordy, x_train_scaled, other_marker_array_sub), file)

    ## TODO save scaler here
    with open(f'./{sample_id}_scaler_refine_level1.pickle', 'wb') as file:
        pickle.dump(CODEX_HIST_scaler, file)

    kf = KFold(n_splits=kfold, random_state=123, shuffle=True)
    fold_metrics = []
    dataset = TensorDataset(torch.tensor(x_train_scaled).to(torch.float32), torch.tensor(y_train).to(torch.float32))

    # Set global random seed
    set_global_seed(123)

    ## set up optimizer and loss
    if selected_model is None:
        model = MLP.Model(n_inp=x_train.shape[-1], n_out=y_train.shape[-1], hidden_dims=hidden_dims, activations=activations, residual=residual, norm=norm, out_activation=out_act, include_tanhshrink=include_tanhshrink)
    elif selected_model == 'lite':
        model = MLP.Model_lite(n_inp=x_train.shape[-1], n_out=y_train.shape[-1], hidden_dims=hidden_dims, activations=activations, out_activation=out_act)
    elif selected_model == 'lite_shrink':
        model = MLP.Model_lite_shrink(n_inp=x_train.shape[-1], n_out=y_train.shape[-1], hidden_dims=hidden_dims, activations=activations, out_activation=out_act)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=False)

        runner = CustomRunner()

        runner.train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loaders={"train": train_loader, "valid": val_loader},
            num_epochs=num_epochs,
            # callbacks=callbacks_list,
            logdir=save_model_dir,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            seed=123
        )

        fold_metrics.append(runner.loader_metrics["loss"])