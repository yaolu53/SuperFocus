import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np
from collections import Counter
from scipy import sparse
import anndata as ad
import pickle
import matplotlib.pyplot as plt
import torch
from src.model_training import MLP
import anndata as ad

def post_procesing(model_dir=None, hidden_dims=[2048, 2048, 1024, 1024], activations=None, residual=False, norm=False, include_tanhshrink=False, data_dir=None, gene_list=None, regress_out=False, resolutions=[0.6, 0.8],
                   refine_factor=1, sample_id=None, radius=116.8):
    with open(data_dir, 'rb') as file:
        x_train, y_train, train_coordx, train_coordy, x_train_scaled = pickle.load(file)

    model = MLP.Model(x_train_scaled.shape[-1], y_train.shape[-1], hidden_dims, activations, residual, norm,
                      include_tanhshrink)
    state_dict = torch.load(f'./logs_{sample_id}_repdoduce/checkpoints/model.best.pth',
                            map_location=torch.device('cpu'))  # Load on CPU
    model.load_state_dict(state_dict)

    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(x_train_scaled).to(torch.float32))

    predicted_gene = np.array(output) * x_train[:, -9][:, None]

    ## Create adata object
    adata = ad.AnnData(predicted_gene)  # ad.AnnData(sparse.csc_matrix(predicted_gene))

    # adata.obs_names = list(dbit_pd.index)
    adata.var_names = gene_list  # list(dbit_pd.columns)

    adata.obs['x'] = train_coordx
    adata.obs['y'] = train_coordy
    adata.obs['section_id'] = [sample_id] * adata.n_obs

    adata.layers['counts'] = predicted_gene.copy()

    sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=True)

    if regress_out:
        # regress out unwanted variables
        sc.pp.regress_out(adata, ['total_counts'])

    # scale data, clip values exceeding standard deviation 10.
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, svd_solver='arpack')

    sc.pp.neighbors(adata, n_neighbors=50, n_pcs=20, use_rep='X_pca', random_state=0)
    sc.tl.umap(adata, random_state=0)

    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_res_{res}", random_state=0, flavor='igraph', n_iterations=2)

    adata.uns = {}
    adata.uns['spatial'] = {}
    adata.uns['spatial'][sample_id] = {}
    adata.uns['spatial'][sample_id]['images'] = {}
    # adata_query.uns['spatial'][sample_id]['images']['lowres'] = imageLN_flip # rotated_imageLN
    adata.uns['spatial'][sample_id]['scalefactors'] = {}
    adata.uns['spatial'][sample_id]['scalefactors']['tissue_hires_scalef'] = 0.13636674  # 78.5
    adata.uns['spatial'][sample_id]['scalefactors'][
        'spot_diameter_fullres'] = radius * 2. / refine_factor  # 127.8358082731091 #2.18
    adata.uns['spatial'][sample_id]['scalefactors']['fiducial_diameter_fullres'] = 1204.802166311371
    adata.uns['spatial'][sample_id]['scalefactors']['tissue_lowres_scalef'] = 0.013636674  # 78.5
    adata.obsm['spatial'] = np.array([adata.obs['x'], adata.obs['y']]).T
    adata.obs['sample_id'] = [sample_id] * adata.n_obs

    return adata



