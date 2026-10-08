# Model Training Toolkit

SuperFocus training pipeline for spatial ATAC-seq and CUT&Tag tasks.
Provides cross-validated training across three predefined model sizes,
with checkpointing, save/load, and inference utilities.

This work is related to the manuscript at
[https://doi.org/10.64898/2025.12.26.696575](https://doi.org/10.64898/2025.12.26.696575).
Code to reproduce the results in the manuscript is in the `tutorials` folder.

## Installation of SuperFocus （No AI access to the codebase)

Requires SuperFocus python environment and Linux x86-64 OS

Place `install_superfocus.sh` in the same directory as `SuperFocus_final_environment.yml`
(conda and git must be available, e.g. `module load miniconda` on an HPC cluster) and install with:

```bash
bash install_superfocus.sh
```

The script creates the conda environment from the yml, installs STalign, and pins
torch 2.0.1+cu118, torchvision 0.15.2+cu118, numpy 1.26.4 and setuptools 69.1.1.
It stops on the first error, checks the pinned versions at the end, and writes a log to
`install_superfocus_<timestamp>.log`. When it finishes it prints the `conda activate`
command for the new environment.

Optional arguments:

```bash
bash install_superfocus.sh -n my_env                  # use a different environment name
bash install_superfocus.sh -p /path/to/envs/my_env    # install to a prefix path
bash install_superfocus.sh -f /path/to/environment.yml # yml in another location
bash install_superfocus.sh --force                    # remove an existing environment and rebuild
```

Then place the src folder we provide in your directory (e.g. ./SuperFocus) and add it to the sys.path using the code below

```python
import sys
import os

# Get the absolute path to the directory containing the script to be imported
script_dir = os.path.abspath("./SuperFocus")

# Add the directory to sys.path
sys.path.append(script_dir)
```

## Installation of CellVit++

Please go to the github of CellVit++ below for installation instruction

[CellVit++](https://github.com/tio-ikim/cellvit-plus-plus)

CellVit++ needs GPU to analyze, please read the requirement on github carefully.
Once you install the CellVit++ environment, 
you can use it to analyze the image (svs format preferred). 
The wsi_properties does not need if you have an svs file. 
For tif file, you will need to convert it to tiled tif first.
We provide the a script for your reference below.

```bash
module load miniconda
conda activate cellvit_env

python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --outdir ./atlasX/Ellis457_Case2_1_tiled/compression \
  --geojson \
  --graph \
  process_wsi \
  --wsi_path ./atlasX/Ellis457_Case2_1_fixed.tif \
  --wsi_properties "{\"slide_mpp\": 0.5, \"magnification\": 20}"

```

## Installation of Snapatac2

We used Snapatac2, a python package, to process fragments.tsv.gz files from spatial ATAC-seq and CUT&Tag. 
It will generate a tile matrix. If you want to use Snapatac2, you can install the environment by the 
following ways:

```bash
conda env create -f snapatac2.yml
```

## Generate model inputs

To feed into SuperFocus model, you need to create an anndata object from the tile matrix.
Each row is the spot and each column is a bin on the chromosome. 
The anndata object must have the following structures:

1. in 'obs', 'array_col', 'array_row', 'x', 'y' must present. 
'x' and 'y' are the coordinates of the centroids of each spot on the H&E image.
2. in 'obsm', 'spatial' must present. It is a 2 by n_obs numpy array, 
with the first column to be 'x' and second to be 'y'.
3. in 'layers', 'counts' must present. 'counts' is the raw tile matrix.

## Train the model

Training has two steps. `training_orig` trains the model at the original spot resolution.
`training` then starts from a checkpoint saved by `training_orig` and continues training
at a refined resolution.

### Step 1: train at the original resolution

Example call:

```python
training_orig(
    adata=adata_sub,
    pt_model_dir=f'../CellViT/maldi_mousebrain/preprocessing/{he_id}/{he_id}/cell_detection',
    radius=564./2,
    feature_list=list(adata_sub.var_names),
    save_model_dir=f'./logs_{sample_id}_fmp',
    hidden_dims=[1024, 1024, 512, 512],
    activations=None,
    residual=False,
    norm=False,
    out_act=None,
    include_tanhshrink=False,
    sample_id=sample_id,
    training_mode=None,
    selected_model='lite_shrink',
    kfold=5,
    cell_embed_dir=f'./{he_id}_fmp_cellvitplusplus_embed.pickle',
    sort_way='col_row',
    datatype='dbit',
)
```

#### Parameters

Data inputs

- `adata`: anndata object with spots as rows and features as columns, in the format described in "Generate model inputs".
- `feature_list`: names of the features to predict, e.g. `list(adata.var_names)` to predict all features.
- `pt_model_dir`: path to the CellVit++ `cell_detection` output folder of the matching H&E image. The code will summarise the cell embedding features from CellVit++. This will be ignored if `cell_embed_dir` is provided.
- `cell_embed_dir`: path to the pickle file of per-cell embeddings. This could be cell embeddings from any digital pathology foundation models.
- `radius`: spot radius in pixels of the H&E image, e.g. `564./2` for a spot width of 564 pixels.
- `datatype`: data type of the sample, e.g. `'dbit'`, `'Visium'`. We have special handling for `'dbit'` data.
- `sort_way`: ordering of the spots, e.g. `'col_row'` to order by `array_col` and then `array_row`.
- `sample_id`: sample identifier used to name the outputs.

Model architecture

- `selected_model`: which of the predefined model sizes to train, e.g. `'lite_shrink'`. We recommend to use `'lite_shrink'`, which has robust performance across various spatial-omics samples. However, you can have your own model settings by set `selected_model` to `None` and tune model architecture parameters below.
- `hidden_dims`: widths of the hidden layers, e.g. `[1024, 1024, 512, 512]` for four hidden layers.
- `activations`: activation function of the hidden layers. `None` uses the default.
- `residual`: whether to use residual connections between hidden layers.
- `norm`: whether to add normalization layers.
- `out_act`: activation function of the output layer. `None` gives a linear output.
- `include_tanhshrink`: whether to add a Tanhshrink activation.

Training

- `kfold`: number of cross-validation folds.
- `training_mode`: training variant. `None` uses the default.
- `save_model_dir`: folder where the model checkpoints and logs for each fold are saved.

### Step 2: train at the refined resolution

Example call:

```python
training(
    adata=adata_sub,
    pt_model_dir=f'../CellViT/maldi_mousebrain/preprocessing/{he_id}/{he_id}/cell_detection',
    radius=564./2,
    refine_factor=2,
    feature_pd_dir=f'./{sample_id}_gene_exp_pd.pickle',
    trained_model_dir=f'./logs_{sample_id}_fmp/fold_4/checkpoints/model.best.pth',
    save_model_dir=f'./logs_{sample_id}_fmp_2x',
    hidden_dims=[1024, 1024, 512, 512],
    activations=None,
    residual=False,
    norm=False,
    out_act=None,
    include_tanhshrink=False,
    sample_id=sample_id,
    training_mode=None,
    selected_model='lite_shrink',
    kfold=5,
    cell_embed_dir=f'./{he_id}_fmp_cellvitplusplus_embed.pickle',
    lr=1e-7,
    step_size=25,
    gamma=0.5,
    num_epochs=50,
    sort_way='col_row',
    random_state=42,
    datatype='dbit',
    max_fold=1,
)
```

#### Additional parameters

The parameters shared with `training_orig` have the same meaning. The model architecture
parameters should match the ones used in step 1, because the model is loaded from its checkpoint.

- `radius`: spot radius in original pixels of the H&E image, same as the paramater in `training_orig`.
- `refine_factor`: factor by which the spot resolution is refined, e.g. `2` for 2x.
- `trained_model_dir`: path to the checkpoint from step 1 to start from.
  We usually use the last fold (`fold_4/checkpoints/model.best.pth` when `kfold=5`),
  but the choice depends on the predictions of the models, so use another fold if its predictions are better.
- `feature_pd_dir`: path to the pickle file of the feature table (pandas dataframe).
- `save_model_dir`: folder where the refined model checkpoints and logs are saved.
  Use a different folder from step 1 so that the step 1 checkpoints are not overwritten.
- `lr`: learning rate.
- `step_size`, `gamma`: learning rate schedule. The learning rate is multiplied by `gamma` every `step_size` epochs.
- `num_epochs`: number of training epochs.
- `random_state`: random seed.
- `max_fold`: maximum number of folds to train, e.g. `1` to train only one of the `kfold` folds. We recommend set `max_fold` to be `1` to avoid overfitting during refinement.
