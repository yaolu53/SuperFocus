import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import torch
import plotly
import requests
import pickle
import json
import scanpy as sc
import anndata as ad
from scipy.sparse.linalg import svds


def cell_positions(pt_model_dir):
    cell_json_dir = pt_model_dir + '/cells.json'
    with open(cell_json_dir) as f:
        cell_data = json.load(f)
    cell_centroid_0 = []
    cell_centroid_1 = []

    for cell in cell_data['cells']:
        cell_centroid_0.append(cell['centroid'][0])
        cell_centroid_1.append(cell['centroid'][1])

    return cell_centroid_0, cell_centroid_1


def original_spots_coords(adata, radius: float, sort_way='row_col'):
    array_obs_vars = {'array_row', 'array_col'}
    if array_obs_vars.issubset(adata.obs.columns):
        visium_coords = pd.DataFrame({'Barcode': adata.obs_names, 'array_row': adata.obs['array_row'],
                                      'array_col': adata.obs['array_col'],
                                      'pxl_row_in_fullres': adata.obsm['spatial'][:, 1],
                                      'pxl_col_in_fullres': adata.obsm['spatial'][:, 0]})
        if sort_way == 'row_col':
            visium_coords = visium_coords.sort_values(['array_row', 'array_col'])
        elif sort_way == 'col_row':
            visium_coords = visium_coords.sort_values(['array_col', 'array_row'])
        visium_coords = visium_coords.reset_index(drop=True)
    else:
        visium_coords = pd.DataFrame({'Barcode': adata.obs_names, 'pxl_row_in_fullres': adata.obsm['spatial'][:, 1],
                                      'pxl_col_in_fullres': adata.obsm['spatial'][:, 0]})

    visium_coords['lower_x'] = visium_coords['pxl_col_in_fullres'] - radius
    visium_coords['upper_x'] = visium_coords['pxl_col_in_fullres'] + radius
    visium_coords['lower_y'] = visium_coords['pxl_row_in_fullres'] - radius
    visium_coords['upper_y'] = visium_coords['pxl_row_in_fullres'] + radius

    visium_coords['lower_x_n1'] = visium_coords['pxl_col_in_fullres'] - radius - 2 * radius
    visium_coords['upper_x_n1'] = visium_coords['pxl_col_in_fullres'] + radius - 2 * radius
    visium_coords['lower_y_n1'] = visium_coords['pxl_row_in_fullres'] - radius + 2 * radius
    visium_coords['upper_y_n1'] = visium_coords['pxl_row_in_fullres'] + radius + 2 * radius

    visium_coords['lower_x_n2'] = visium_coords['pxl_col_in_fullres'] - radius
    visium_coords['upper_x_n2'] = visium_coords['pxl_col_in_fullres'] + radius
    visium_coords['lower_y_n2'] = visium_coords['pxl_row_in_fullres'] - radius + 2 * radius
    visium_coords['upper_y_n2'] = visium_coords['pxl_row_in_fullres'] + radius + 2 * radius

    visium_coords['lower_x_n3'] = visium_coords['pxl_col_in_fullres'] - radius + 2 * radius
    visium_coords['upper_x_n3'] = visium_coords['pxl_col_in_fullres'] + radius + 2 * radius
    visium_coords['lower_y_n3'] = visium_coords['pxl_row_in_fullres'] - radius + 2 * radius
    visium_coords['upper_y_n3'] = visium_coords['pxl_row_in_fullres'] + radius + 2 * radius

    visium_coords['lower_x_n4'] = visium_coords['pxl_col_in_fullres'] - radius - 2 * radius
    visium_coords['upper_x_n4'] = visium_coords['pxl_col_in_fullres'] + radius - 2 * radius
    visium_coords['lower_y_n4'] = visium_coords['pxl_row_in_fullres'] - radius
    visium_coords['upper_y_n4'] = visium_coords['pxl_row_in_fullres'] + radius

    visium_coords['lower_x_n5'] = visium_coords['pxl_col_in_fullres'] - radius + 2 * radius
    visium_coords['upper_x_n5'] = visium_coords['pxl_col_in_fullres'] + radius + 2 * radius
    visium_coords['lower_y_n5'] = visium_coords['pxl_row_in_fullres'] - radius
    visium_coords['upper_y_n5'] = visium_coords['pxl_row_in_fullres'] + radius

    visium_coords['lower_x_n6'] = visium_coords['pxl_col_in_fullres'] - radius - 2 * radius
    visium_coords['upper_x_n6'] = visium_coords['pxl_col_in_fullres'] + radius - 2 * radius
    visium_coords['lower_y_n6'] = visium_coords['pxl_row_in_fullres'] - radius - 2 * radius
    visium_coords['upper_y_n6'] = visium_coords['pxl_row_in_fullres'] + radius - 2 * radius

    visium_coords['lower_x_n7'] = visium_coords['pxl_col_in_fullres'] - radius
    visium_coords['upper_x_n7'] = visium_coords['pxl_col_in_fullres'] + radius
    visium_coords['lower_y_n7'] = visium_coords['pxl_row_in_fullres'] - radius - 2 * radius
    visium_coords['upper_y_n7'] = visium_coords['pxl_row_in_fullres'] + radius - 2 * radius

    visium_coords['lower_x_n8'] = visium_coords['pxl_col_in_fullres'] - radius + 2 * radius
    visium_coords['upper_x_n8'] = visium_coords['pxl_col_in_fullres'] + radius + 2 * radius
    visium_coords['lower_y_n8'] = visium_coords['pxl_row_in_fullres'] - radius - 2 * radius
    visium_coords['upper_y_n8'] = visium_coords['pxl_row_in_fullres'] + radius - 2 * radius

    return visium_coords


def generate_grid(side_length, num_points):
    """
    Generates a grid of points within a square.

    Args:
      side_length: The length of the side of the square.
      num_points: The number of points along each axis of the grid.

    Returns:
      A NumPy array of shape (num_points, num_points, 2) containing the coordinates
      of the grid points.
    """
    x = np.linspace(0, side_length, num_points)
    y = np.linspace(0, side_length, num_points)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv, yv], axis=-1)


def refined_spots_coords(adata, radius: float, refine_factor: int, sort_way='row_col'):
    array_obs_vars = {'array_row', 'array_col'}
    if array_obs_vars.issubset(adata.obs.columns):
        visium_coords = pd.DataFrame({'Barcode': adata.obs_names, 'array_row': adata.obs['array_row'],
                                      'array_col': adata.obs['array_col'],
                                      'pxl_row_in_fullres': adata.obsm['spatial'][:, 1],
                                      'pxl_col_in_fullres': adata.obsm['spatial'][:, 0]})
        if sort_way == 'row_col':
            visium_coords = visium_coords.sort_values(['array_row', 'array_col'])
        elif sort_way == 'col_row':
            visium_coords = visium_coords.sort_values(['array_col', 'array_row'])
        visium_coords = visium_coords.reset_index(drop=True)
    else:
        visium_coords = pd.DataFrame({'Barcode': adata.obs_names, 'pxl_row_in_fullres': adata.obsm['spatial'][:, 1],
                                      'pxl_col_in_fullres': adata.obsm['spatial'][:, 0]})

    visium_coords['lower_x'] = visium_coords['pxl_col_in_fullres'] - radius
    visium_coords['upper_x'] = visium_coords['pxl_col_in_fullres'] + radius
    visium_coords['lower_y'] = visium_coords['pxl_row_in_fullres'] - radius
    visium_coords['upper_y'] = visium_coords['pxl_row_in_fullres'] + radius

    x_grid = generate_grid(radius * 2, refine_factor + 1)[:, :, 0]
    y_grid = generate_grid(radius * 2, refine_factor + 1)[:, :, 1]

    grid_coords = np.zeros((refine_factor ** 2, 4))
    for j in range(refine_factor):
        for k in range(refine_factor):
            grid_coords[j * refine_factor + k, 0] = x_grid[j, k]
            grid_coords[j * refine_factor + k, 1] = x_grid[j, k + 1]
            grid_coords[j * refine_factor + k, 2] = y_grid[j, k]
            grid_coords[j * refine_factor + k, 3] = y_grid[j + 1, k]

    visium_refined_coords = np.zeros((visium_coords.shape[0] * refine_factor ** 2, 4))

    for i in range(visium_coords.shape[0]):
        visium_refined_coords[(i * grid_coords.shape[0]):((i + 1) * grid_coords.shape[0]), :2] = grid_coords[:, :2] + \
                                                                                                 visium_coords.loc[
                                                                                                     i, 'lower_x']
        visium_refined_coords[(i * grid_coords.shape[0]):((i + 1) * grid_coords.shape[0]), 2:] = grid_coords[:, 2:] + \
                                                                                                 visium_coords.loc[
                                                                                                     i, 'lower_y']

    visium_coords_repeat = visium_coords.loc[visium_coords.index.repeat(refine_factor ** 2)].reset_index(drop=True)

    visium_coords_repeat['lower_x_refined'] = visium_refined_coords[:, 0]
    visium_coords_repeat['upper_x_refined'] = visium_refined_coords[:, 1]
    visium_coords_repeat['lower_y_refined'] = visium_refined_coords[:, 2]
    visium_coords_repeat['upper_y_refined'] = visium_refined_coords[:, 3]

    radius /= refine_factor
    visium_coords_repeat['lower_x_refined_n1'] = visium_coords_repeat['lower_x_refined'] - 2 * radius
    visium_coords_repeat['upper_x_refined_n1'] = visium_coords_repeat['lower_x_refined']
    visium_coords_repeat['lower_y_refined_n1'] = visium_coords_repeat['lower_y_refined'] - 2 * radius
    visium_coords_repeat['upper_y_refined_n1'] = visium_coords_repeat['lower_y_refined']

    visium_coords_repeat['lower_x_refined_n2'] = visium_coords_repeat['lower_x_refined'] - 2 * radius
    visium_coords_repeat['upper_x_refined_n2'] = visium_coords_repeat['lower_x_refined']
    visium_coords_repeat['lower_y_refined_n2'] = visium_coords_repeat['lower_y_refined']
    visium_coords_repeat['upper_y_refined_n2'] = visium_coords_repeat['upper_y_refined']

    visium_coords_repeat['lower_x_refined_n3'] = visium_coords_repeat['lower_x_refined'] - 2 * radius
    visium_coords_repeat['upper_x_refined_n3'] = visium_coords_repeat['lower_x_refined']
    visium_coords_repeat['lower_y_refined_n3'] = visium_coords_repeat['upper_y_refined']
    visium_coords_repeat['upper_y_refined_n3'] = visium_coords_repeat['upper_y_refined'] + 2 * radius

    visium_coords_repeat['lower_x_refined_n4'] = visium_coords_repeat['lower_x_refined']
    visium_coords_repeat['upper_x_refined_n4'] = visium_coords_repeat['upper_x_refined']
    visium_coords_repeat['lower_y_refined_n4'] = visium_coords_repeat['lower_y_refined'] - 2 * radius
    visium_coords_repeat['upper_y_refined_n4'] = visium_coords_repeat['lower_y_refined']

    visium_coords_repeat['lower_x_refined_n5'] = visium_coords_repeat['lower_x_refined']
    visium_coords_repeat['upper_x_refined_n5'] = visium_coords_repeat['upper_x_refined']
    visium_coords_repeat['lower_y_refined_n5'] = visium_coords_repeat['upper_y_refined']
    visium_coords_repeat['upper_y_refined_n5'] = visium_coords_repeat['upper_y_refined'] + 2 * radius

    visium_coords_repeat['lower_x_refined_n6'] = visium_coords_repeat['upper_x_refined']
    visium_coords_repeat['upper_x_refined_n6'] = visium_coords_repeat['upper_x_refined'] + 2 * radius
    visium_coords_repeat['lower_y_refined_n6'] = visium_coords_repeat['lower_y_refined'] - 2 * radius
    visium_coords_repeat['upper_y_refined_n6'] = visium_coords_repeat['lower_y_refined']

    visium_coords_repeat['lower_x_refined_n7'] = visium_coords_repeat['upper_x_refined']
    visium_coords_repeat['upper_x_refined_n7'] = visium_coords_repeat['upper_x_refined'] + 2 * radius
    visium_coords_repeat['lower_y_refined_n7'] = visium_coords_repeat['lower_y_refined']
    visium_coords_repeat['upper_y_refined_n7'] = visium_coords_repeat['upper_y_refined']

    visium_coords_repeat['lower_x_refined_n8'] = visium_coords_repeat['upper_x_refined']
    visium_coords_repeat['upper_x_refined_n8'] = visium_coords_repeat['upper_x_refined'] + 2 * radius
    visium_coords_repeat['lower_y_refined_n8'] = visium_coords_repeat['upper_y_refined']
    visium_coords_repeat['upper_y_refined_n8'] = visium_coords_repeat['upper_y_refined'] + 2 * radius

    return visium_coords_repeat


def standardize(embeddings):
    mean = embeddings.mean(dim=0, keepdim=True)
    std = embeddings.std(dim=0, keepdim=True)
    return (embeddings - mean) / (std + 1e-6)


def process_embedding(pt_model_dir):
    cell_pt_dir = pt_model_dir + '/cells.pt'

    model = torch.load(cell_pt_dir)
    matched_embed = model.x
    standardized_embeddings = standardize(matched_embed)
    final_embed_norm = np.array(standardized_embeddings)

    ## dimensional reduction using SVD
    #u, s, vh = svds(final_embed_norm, final_embed_norm.shape[1] - 1)

    n_component = 200
    u, s, vh = svds(final_embed_norm, n_component, random_state=123)
    final_embed_norm_recon = u @ np.diag(s)

    final_embed_norm_recon_df = pd.DataFrame(final_embed_norm_recon)

    return final_embed_norm_recon_df


def cells_counts(cell_df: pd.DataFrame, visium_coords):  # visium_coords: lower_x, upper_x, lower_y, upper_y

    if not isinstance(visium_coords, np.ndarray):
        raise ValueError("Coordinates input must be a numpy array")

    x_lower = np.floor(np.min(visium_coords[:, 0])).astype(int)
    x_upper = np.ceil(np.max(visium_coords[:, 1])).astype(int)

    y_lower = np.floor(np.min(visium_coords[:, 2])).astype(int)
    y_upper = np.ceil(np.max(visium_coords[:, 3])).astype(int)

    xI_cells = np.array(cell_df['x'])
    yI_cells = np.array(cell_df['y'])

    cell_df_sub = cell_df.loc[
                  (xI_cells >= x_lower) & (xI_cells <= x_upper) & (yI_cells >= y_lower) & (yI_cells <= y_upper), :]

    xI_cells_sub = np.array(cell_df_sub['x'])
    yI_cells_sub = np.array(cell_df_sub['y'])

    cell_mat = np.zeros((visium_coords.shape[0], 9))

    for j in range(9):
        for i in range(visium_coords.shape[0]):
            cell_df_sub_sub = cell_df_sub.loc[(xI_cells_sub >= visium_coords[i, 0 + j * 4]) & (
                        xI_cells_sub <= visium_coords[i, 1 + j * 4]) & (yI_cells_sub >= visium_coords[i, 2 + j * 4]) & (
                                                          yI_cells_sub <= visium_coords[i, 3 + j * 4]), :]
            cell_mat[i, j] = cell_df_sub_sub.shape[0]
    return cell_mat


def cells_markers(cell_df: pd.DataFrame, visium_coords):  # visium_coords: lower_x, upper_x, lower_y, upper_y

    if not isinstance(visium_coords, np.ndarray):
        raise ValueError("Coordinates input must be a numpy array")

    x_lower = np.floor(np.min(visium_coords[:, 0])).astype(int)
    x_upper = np.ceil(np.max(visium_coords[:, 1])).astype(int)

    y_lower = np.floor(np.min(visium_coords[:, 2])).astype(int)
    y_upper = np.ceil(np.max(visium_coords[:, 3])).astype(int)

    xI_cells = np.array(cell_df['x'])
    yI_cells = np.array(cell_df['y'])

    cell_df_sub = cell_df.loc[
                  (xI_cells >= x_lower) & (xI_cells <= x_upper) & (yI_cells >= y_lower) & (yI_cells <= y_upper), :]

    xI_cells_sub = np.array(cell_df_sub['x'])
    yI_cells_sub = np.array(cell_df_sub['y'])

    cell_df_sub = cell_df_sub.iloc[:, :-2]

    cell_mat = np.zeros((visium_coords.shape[0], 200 * 9 * 2))

    for j in range(9):
        for i in range(visium_coords.shape[0]):
            cell_df_sub_sub = cell_df_sub.loc[
                (xI_cells_sub >= visium_coords[i, 0 + j * 4]) & (xI_cells_sub <= visium_coords[i, 1 + j * 4]) & (
                            yI_cells_sub >= visium_coords[i, 2 + j * 4]) & (
                            yI_cells_sub <= visium_coords[i, 3 + j * 4]),]
            if cell_df_sub_sub.shape[0] != 0:
                min_markers = np.array(cell_df_sub_sub.min(axis=0))
                max_markers = np.array(cell_df_sub_sub.max(axis=0))
                cell_mat[i, (400 * j):(400 * j + 200)] = min_markers
                cell_mat[i, (400 * j + 200):(400 * (j + 1))] = max_markers
    return cell_mat


def process_image_features(adata, pt_model_dir: str, radius: float, refine_factor=1, cell_embed_dir=None, sort_way='row_col'):

    if cell_embed_dir is None:
        cell_embed_df = process_embedding(pt_model_dir)
        cell_centroid_0, cell_centroid_1 = cell_positions(pt_model_dir)
        cell_embed_df['x'] = cell_centroid_0
        cell_embed_df['y'] = cell_centroid_1
    else:
        with open(cell_embed_dir, 'rb') as handle:
            cell_embed_df = pickle.load(handle)
        cell_centroid_0 = cell_embed_df['x']
        cell_centroid_1 = cell_embed_df['y']

    cell_pos_df = pd.DataFrame({'x': cell_centroid_0, 'y': cell_centroid_1})

    if refine_factor == 1:
        visium_coords = original_spots_coords(adata, radius, sort_way)
        visium_coords_array = np.array(visium_coords.iloc[:, visium_coords.columns.get_loc('lower_x'):])
        visium_cell_counts = cells_counts(cell_pos_df, visium_coords_array)
        visium_cell_markers = cells_markers(cell_embed_df, visium_coords_array)

        return visium_cell_markers, visium_cell_counts, visium_coords, cell_embed_df
    else:
        visium_coords_repeat = refined_spots_coords(adata, radius, refine_factor, sort_way)
        visium_coords_array = np.array(
            visium_coords_repeat.iloc[:, visium_coords_repeat.columns.get_loc('lower_x_refined'):])
        visium_cell_counts = cells_counts(cell_pos_df, visium_coords_array)
        visium_cell_markers = cells_markers(cell_embed_df, visium_coords_array)

        return visium_cell_markers, visium_cell_counts, visium_coords_repeat, cell_embed_df


def whole_image_all_cells_features(pt_model_dir: str, radius: float, cell_embed_dir=None):
    if cell_embed_dir is None:
        cell_embed_df = process_embedding(pt_model_dir)
        cell_centroid_0, cell_centroid_1 = cell_positions(pt_model_dir)
        cell_embed_df['x'] = cell_centroid_0
        cell_embed_df['y'] = cell_centroid_1
    else:
        with open(cell_embed_dir, 'rb') as handle:
            cell_embed_df = pickle.load(handle)
        cell_centroid_0 = cell_embed_df['x']
        cell_centroid_1 = cell_embed_df['y']
    #cell_centroid_0, cell_centroid_1 = cell_positions(pt_model_dir)
    cell_pos_df = pd.DataFrame({'x': cell_centroid_0, 'y': cell_centroid_1})

    #cell_embed_df = process_embedding(pt_model_dir)
    #cell_embed_df['x'] = cell_centroid_0
    #cell_embed_df['y'] = cell_centroid_1

    cell_coords = pd.DataFrame({'Barcode': range(len(cell_centroid_0)), 'pxl_row_in_fullres': cell_centroid_1,
                                'pxl_col_in_fullres': cell_centroid_0})

    cell_coords['lower_x'] = cell_coords['pxl_col_in_fullres'] - radius
    cell_coords['upper_x'] = cell_coords['pxl_col_in_fullres'] + radius
    cell_coords['lower_y'] = cell_coords['pxl_row_in_fullres'] - radius
    cell_coords['upper_y'] = cell_coords['pxl_row_in_fullres'] + radius

    cell_coords['lower_x_n1'] = cell_coords['pxl_col_in_fullres'] - radius - 2 * radius
    cell_coords['upper_x_n1'] = cell_coords['pxl_col_in_fullres'] + radius - 2 * radius
    cell_coords['lower_y_n1'] = cell_coords['pxl_row_in_fullres'] - radius + 2 * radius
    cell_coords['upper_y_n1'] = cell_coords['pxl_row_in_fullres'] + radius + 2 * radius

    cell_coords['lower_x_n2'] = cell_coords['pxl_col_in_fullres'] - radius
    cell_coords['upper_x_n2'] = cell_coords['pxl_col_in_fullres'] + radius
    cell_coords['lower_y_n2'] = cell_coords['pxl_row_in_fullres'] - radius + 2 * radius
    cell_coords['upper_y_n2'] = cell_coords['pxl_row_in_fullres'] + radius + 2 * radius

    cell_coords['lower_x_n3'] = cell_coords['pxl_col_in_fullres'] - radius + 2 * radius
    cell_coords['upper_x_n3'] = cell_coords['pxl_col_in_fullres'] + radius + 2 * radius
    cell_coords['lower_y_n3'] = cell_coords['pxl_row_in_fullres'] - radius + 2 * radius
    cell_coords['upper_y_n3'] = cell_coords['pxl_row_in_fullres'] + radius + 2 * radius

    cell_coords['lower_x_n4'] = cell_coords['pxl_col_in_fullres'] - radius - 2 * radius
    cell_coords['upper_x_n4'] = cell_coords['pxl_col_in_fullres'] + radius - 2 * radius
    cell_coords['lower_y_n4'] = cell_coords['pxl_row_in_fullres'] - radius
    cell_coords['upper_y_n4'] = cell_coords['pxl_row_in_fullres'] + radius

    cell_coords['lower_x_n5'] = cell_coords['pxl_col_in_fullres'] - radius + 2 * radius
    cell_coords['upper_x_n5'] = cell_coords['pxl_col_in_fullres'] + radius + 2 * radius
    cell_coords['lower_y_n5'] = cell_coords['pxl_row_in_fullres'] - radius
    cell_coords['upper_y_n5'] = cell_coords['pxl_row_in_fullres'] + radius

    cell_coords['lower_x_n6'] = cell_coords['pxl_col_in_fullres'] - radius - 2 * radius
    cell_coords['upper_x_n6'] = cell_coords['pxl_col_in_fullres'] + radius - 2 * radius
    cell_coords['lower_y_n6'] = cell_coords['pxl_row_in_fullres'] - radius - 2 * radius
    cell_coords['upper_y_n6'] = cell_coords['pxl_row_in_fullres'] + radius - 2 * radius

    cell_coords['lower_x_n7'] = cell_coords['pxl_col_in_fullres'] - radius
    cell_coords['upper_x_n7'] = cell_coords['pxl_col_in_fullres'] + radius
    cell_coords['lower_y_n7'] = cell_coords['pxl_row_in_fullres'] - radius - 2 * radius
    cell_coords['upper_y_n7'] = cell_coords['pxl_row_in_fullres'] + radius - 2 * radius

    cell_coords['lower_x_n8'] = cell_coords['pxl_col_in_fullres'] - radius + 2 * radius
    cell_coords['upper_x_n8'] = cell_coords['pxl_col_in_fullres'] + radius + 2 * radius
    cell_coords['lower_y_n8'] = cell_coords['pxl_row_in_fullres'] - radius - 2 * radius
    cell_coords['upper_y_n8'] = cell_coords['pxl_row_in_fullres'] + radius - 2 * radius

    visium_coords_array = np.array(cell_coords.iloc[:, cell_coords.columns.get_loc('lower_x'):])
    visium_cell_counts = cells_counts(cell_pos_df, visium_coords_array)
    visium_cell_markers = cells_markers(cell_embed_df, visium_coords_array)

    return visium_cell_markers, visium_cell_counts, cell_coords
