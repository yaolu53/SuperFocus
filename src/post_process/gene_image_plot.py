import pandas as pd
import numpy as np
import scanpy as sc
import pickle
import anndata as ad
import os
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageOps


def match_spot(adata, adata_origin, cut_off):
    selected_spots = []
    dist_x = []
    dist_y = []
    for i in range(adata.obsm['spatial'].shape[0]):
        tmp_dist = (adata_origin.obsm['spatial'][:, 0] - adata.obs['x'][i]) ** 2 + (
                    adata_origin.obsm['spatial'][:, 1] - adata.obs['y'][i]) ** 2
        tmp_index = np.where(tmp_dist == np.min(tmp_dist))[0][0]
        selected_spots.append(tmp_index)
        dist_x.append(np.abs(adata_origin.obsm['spatial'][tmp_index, 0] - adata.obs['x'][i]))
        dist_y.append(np.abs(adata_origin.obsm['spatial'][tmp_index, 1] - adata.obs['y'][i]))

    cut_off = cut_off
    matched_spots = (np.array(dist_x) < cut_off) * (np.array(dist_y) < cut_off)
    final_selected_spots = np.array(selected_spots)[matched_spots]

    return matched_spots, final_selected_spots

def add_background_image_to_adata(adata, hires_image_dir, lowres_image_dir, scale_data, sample_id):
    '''this function is to add H&E image to anndata'''
    hires_image = np.array(
        Image.open(hires_image_dir))
    lowres_image = np.array(
        Image.open(lowres_image_dir))

    adata.uns['spatial'] = {}
    adata.uns['spatial'][sample_id] = {}
    adata.uns['spatial'][sample_id]['images'] = {}
    adata.uns['spatial'][sample_id]['images']['hires'] = hires_image
    adata.uns['spatial'][sample_id]['images']['lowres'] = lowres_image
    adata.uns['spatial'][sample_id]['scalefactors'] = scale_data

    return adata



def marker_gene_plot(adata, invert=False, sample_id=None, image_key='hires', marker_gene=None, size=1.0, alpha_image=1.0, vmin=-None, vmax=None, format='pdf'):
    '''the anndata here needs to have H&E image added'''
    if invert:
        he_img = adata.uns['spatial'][sample_id]['images'][image_key]
        pil_he_img = Image.fromarray(he_img)
        invert_he_img = ImageOps.invert(pil_he_img)
        gray_he_img = invert_he_img.convert('LA')
        gray_he_img_array = np.array(gray_he_img)
        gray_he_img_final = np.stack((gray_he_img_array[:, :, 0],) * 3, axis=-1)

        adata.uns['spatial'][sample_id]['images']['invert'] = gray_he_img_final
        adata.uns['spatial'][sample_id]["scalefactors"]['tissue_invert_scalef'] = \
        adata.uns['spatial'][sample_id]["scalefactors"]['tissue_hires_scalef']

        sc.set_figure_params(dpi=400)

        fig = sc.pl.spatial(adata, color=marker_gene, size=size, img_key='invert', alpha_img=alpha_image, legend_loc=None,
                            frameon=False, colorbar_loc=None, title='', show=False, return_fig=True, vmin=vmin, vmax=vmax)
        fig.savefig(f'{sample_id}_{marker_gene}.{format}', format=format, bbox_inches="tight")

