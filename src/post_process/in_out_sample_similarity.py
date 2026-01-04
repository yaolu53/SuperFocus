import pandas as pd
from src.model_training.processing_image_features import cell_positions, process_embedding
import faiss
import numpy as np
from sklearn.decomposition import PCA

def image_features_extract(cell_df, radius, cell_centroid_0=None, cell_centroid_1=None):
    if cell_centroid_0 is None:
        cell_centroid_0 = cell_df['x']
        cell_centroid_1 = cell_df['y']

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

    visium_coords = np.array(cell_coords.iloc[:, cell_coords.columns.get_loc('lower_x'):])

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

    cell_num_mat = np.zeros((visium_coords.shape[0], 9))
    cell_marker_mat = np.zeros((visium_coords.shape[0], 200 * 9 * 2))

    for j in range(9):
        for i in range(visium_coords.shape[0]):
            cell_df_sub_sub = cell_df_sub.loc[(xI_cells_sub >= visium_coords[i, 0 + j * 4]) & (
                        xI_cells_sub <= visium_coords[i, 1 + j * 4]) & (yI_cells_sub >= visium_coords[i, 2 + j * 4]) & (
                                                          yI_cells_sub <= visium_coords[i, 3 + j * 4]), :]
            cell_num_mat[i, j] = cell_df_sub_sub.shape[0]

            if cell_df_sub_sub.shape[0] != 0:
                min_markers = np.array(cell_df_sub_sub.min(axis=0))
                max_markers = np.array(cell_df_sub_sub.max(axis=0))
                cell_marker_mat[i, (400 * j):(400 * j + 200)] = min_markers
                cell_marker_mat[i, (400 * j + 200):(400 * (j + 1))] = max_markers

    return cell_num_mat, cell_marker_mat


def rotate_function(x, y, deg):
    xI_scale = x
    yI_scale = y
    ## rotate DBiT ROI
    theta_deg = deg
    theta0 = (np.pi / 180) * -theta_deg

    # rotation matrix
    # rotates about the origin
    L = np.array([[np.cos(theta0), -np.sin(theta0)],
                  [np.sin(theta0), np.cos(theta0)]])

    source_L = np.matmul(L, np.array([xI_scale, yI_scale]))
    xI_L = source_L[0]
    yI_L = source_L[1]

    # translation matrix
    # effectively makes the rotation about the centroid of I (i.e the means of xI and yI])
    # and also moves the centroid of I to the centroid of J
    T = np.array([np.mean(xI_scale) - np.cos(theta0) * np.mean(xI_scale) + np.sin(theta0) * np.mean(yI_scale),
                  # - (np.mean(xI_scale)-np.mean(xJ)),
                  np.mean(yI_scale) - np.sin(theta0) * np.mean(xI_scale) - np.cos(theta0) * np.mean(
                      yI_scale)])  # - (np.mean(yI_scale)-np.mean(yJ))])

    xI_L_T = xI_L + T[0]  # - 500.
    yI_L_T = yI_L + T[1]  # + 200.

    return xI_L_T, yI_L_T

def in_out_sample_similarity(pt_model_dir, radius, visium_coords, refine_factor, insample_percent, outsample_percent, n_neighbors, sub_cell_index=None):
    '''visium_coords should be a pandas dataframe with pxl_col_in_fullres for x	and pxl_row_in_fullres'''
    '''sub_cell_index should be a list of strings of selected cell indices'''

    if not isinstance(n_neighbors, int) or n_neighbors < 0:
        raise ValueError(f"Expected positive integer, got {n_neighbors}")

    centroid_0, centroid_1 = cell_positions(pt_model_dir=pt_model_dir)
    final_embed_norm_recon_df = process_embedding(pt_model_dir=pt_model_dir)
    final_embed_norm_recon_df['x'] = np.array(centroid_0)
    final_embed_norm_recon_df['y'] = np.array(centroid_1)

    radius = radius
    insample_ind = []
    for i in range(len(cell_centroid_0)):
        if np.sum((np.abs(visium_coords.loc[:, 'pxl_col_in_fullres'] - cell_centroid_0[i]) <= radius) * (
                np.abs(visium_coords.loc[:, 'pxl_row_in_fullres'] - cell_centroid_1[i]) <= radius)):
            insample_ind.append(True)
        else:
            insample_ind.append(False)

    cell_num_mat_0, cell_marker_mat_0 = image_features_extract(final_embed_norm_recon_df, radius/refine_factor)

    x_0 = np.array(final_embed_norm_recon_df['x'])
    y_0 = np.array(final_embed_norm_recon_df['y'])
    x_90, y_90 = rotate_function(x_0, y_0, 90)
    x_180, y_180 = rotate_function(x_0, y_0, 180)
    x_270, y_270 = rotate_function(x_0, y_0, 270)
    x_hp, y_hp = -x_0, y_0
    x_vp, y_vp = x_0, -y_0
    x_hp_90, y_hp_90 = rotate_function(-x_0, y_0, 90)
    x_vp_90, y_vp_90 = rotate_function(x_0, -y_0, 90)

    cell_num_mat_90, cell_marker_mat_90 = image_features_extract(final_embed_norm_recon_df, radius/refine_factor, x_90, y_90)
    cell_num_mat_180, cell_marker_mat_180 = image_features_extract(final_embed_norm_recon_df, radius/refine_factor, x_180, y_180)
    cell_num_mat_270, cell_marker_mat_270 = image_features_extract(final_embed_norm_recon_df, radius/refine_factor, x_270, y_270)
    cell_num_mat_hp, cell_marker_mat_hp = image_features_extract(final_embed_norm_recon_df, radius/refine_factor, x_hp, y_hp)
    cell_num_mat_vp, cell_marker_mat_vp = image_features_extract(final_embed_norm_recon_df, radius/refine_factor, x_vp, y_vp)
    cell_num_mat_hp_90, cell_marker_mat_hp_90 = image_features_extract(final_embed_norm_recon_df, radius/refine_factor, x_hp_90, y_hp_90)
    cell_num_mat_vp_90, cell_marker_mat_vp_90 = image_features_extract(final_embed_norm_recon_df, radius/refine_factor, x_vp_90, y_vp_90)

    cell_features_0 = np.hstack((cell_marker_mat_0, cell_num_mat_0))[insample_ind, :]
    cell_features_90 = np.hstack((cell_marker_mat_90, cell_num_mat_90))[insample_ind, :]
    cell_features_180 = np.hstack((cell_marker_mat_180, cell_num_mat_180))[insample_ind, :]
    cell_features_270 = np.hstack((cell_marker_mat_270, cell_num_mat_270))[insample_ind, :]
    cell_features_hp = np.hstack((cell_marker_mat_hp, cell_num_mat_hp))[insample_ind, :]
    cell_features_vp = np.hstack((cell_marker_mat_vp, cell_num_mat_vp))[insample_ind, :]
    cell_features_hp_90 = np.hstack((cell_marker_mat_hp_90, cell_num_mat_hp_90))[insample_ind, :]
    cell_features_vp_90 = np.hstack((cell_marker_mat_vp_90, cell_num_mat_vp_90))[insample_ind, :]

    rng = np.random.default_rng(seed=42)
    sampled_insample_ind = np.random.choice(cell_features_0.shape[0], size=int(cell_features_0.shape[0] * insample_percent),
                                            replace=False)
    insample_cell_features = np.vstack((cell_features_0[sampled_insample_ind, :],
                                        cell_features_90[sampled_insample_ind, :],
                                        cell_features_180[sampled_insample_ind, :],
                                        cell_features_270[sampled_insample_ind, :],
                                        cell_features_hp[sampled_insample_ind, :],
                                        cell_features_vp[sampled_insample_ind, :],
                                        cell_features_hp_90[sampled_insample_ind, :],
                                        cell_features_vp_90[sampled_insample_ind, :]
                                        ))

    rng = np.random.default_rng(seed=42)
    sampled_outsample_ind = np.random.choice(np.sum(~np.array(insample_ind)),
                                             size=int(np.sum(~np.array(insample_ind)) * outsample_percent), replace=False)
    if sub_cell_index is not None:
        sampled_outsample_ind = np.intersect1d(np.where(~np.array(insample_ind))[0].astype(str), sub_cell_index)
    outsample_cell_features = np.hstack((cell_marker_mat_0, cell_num_mat_0))[~np.array(insample_ind), :]
    outsample_cell_features = outsample_cell_features[sampled_outsample_ind, :]

    outsample_cell_x = np.array(final_embed_norm_recon_df['x'])[~np.array(insample_ind)][sampled_outsample_ind]
    insample_cell_x = np.array(final_embed_norm_recon_df['x'])[np.array(insample_ind)][sampled_insample_ind]
    outsample_cell_y = np.array(final_embed_norm_recon_df['y'])[~np.array(insample_ind)][sampled_outsample_ind]
    insample_cell_y = np.array(final_embed_norm_recon_df['y'])[np.array(insample_ind)][sampled_insample_ind]

    pca = PCA(n_components=200)
    insample_cell_features_reduced = pca.fit_transform(insample_cell_features).astype('float32')
    outsample_cell_features_reduced = pca.transform(outsample_cell_features).astype('float32')

    index = faiss.IndexFlatL2(200)
    index.add(insample_cell_features_reduced)

    distances, indices = index.search(outsample_cell_features_reduced, n_neighbors)

    distances_gmean = np.exp(np.mean(np.log(distances), axis=1))

    credibility_score = 1/np.sqrt(distances_gmean)

    return credibility_score, outsample_cell_x, outsample_cell_y











