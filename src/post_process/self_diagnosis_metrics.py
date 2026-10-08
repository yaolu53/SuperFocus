import pandas as pd
import numpy as np
import scanpy as sc
import pickle
import anndata as ad
import os
import matplotlib.pyplot as plt
#from gene_image_plot import match_spot

def genes_RSSE(counts_est, counts_orig, add_on=0.0, sqrt_root=True):
    sse_list = []
    for i in range(counts_est.shape[1]):
        diff = np.array(counts_orig.iloc[:, i])-np.array(counts_est.iloc[:, i])
        sse = diff.dot(diff)
        if sqrt_root:
            sss = np.sqrt(np.sum(np.array(counts_orig.iloc[:, i]) **2))
        else:
            sss = np.sum(np.array(counts_orig.iloc[:, i]) **2)
        sse_list.append(sse/(sss+add_on))
    return sse_list

def genes_ELS(counts_est, counts_orig, add_on=0.0):
    sparsity_list = []
    for i in range(counts_est.shape[1]):
        l1_norm = np.abs(np.array(counts_est.iloc[:, i]) - np.array(counts_orig.iloc[:, i])).sum()
        l2_norm = np.linalg.norm(np.array(counts_est.iloc[:, i]) - np.array(counts_orig.iloc[:, i]))
        sparsity_list.append(l2_norm/(l1_norm+add_on))
    return sparsity_list

def RSSE_ELS_function(adata_refined, adata_origin, refine_factor, sample_id, training_data_dir):
    '''adata_refined and adata_origin must have original predicted counts stored in layers['counts']'''

    with open(f'{training_data_dir}/{sample_id}_training_data_level{refine_factor}.pickle', 'rb') as f:
        _, _, _, _, _, other_marker_array = pickle.load(f)
    f.close()

    counts = adata_refined.to_df(layer='counts')
    counts['Barcode'] = np.array(other_marker_array.loc[:, 'Barcode'])

    sum_counts = counts.groupby('Barcode').sum()

    counts_origin = adata_origin.to_df(layer='counts')

    counts_origin_sub = counts_origin.loc[sum_counts.index, sum_counts.columns]

    rsse_sum = genes_RSSE(sum_counts, counts_origin_sub)
    els_sum = genes_ELS(sum_counts, counts_origin_sub)

    return rsse_sum, els_sum, sum_counts.columns




