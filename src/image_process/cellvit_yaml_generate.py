import yaml
import os

def yaml_generate(wsi_folder, wsi_name, target_mag, n_processes, overwrite, normalize_stains, min_intersection_ratio):
    wsi_path = wsi_folder+'/'+wsi_name
    wsi_name_no_extension = wsi_name.rsplit(".", 1)[-2]
    output_path = wsi_path +'/preprocessing/'+wsi_name_no_extension
    os.makedirs(output_path, exist_ok=True)
    
    data = {
        "wsi_paths": wsi_path,
        "output_path": output_path,
        "patch_size": 1024,
        "patch_overlap": 6.25,
        "target_mag": target_mag,
        "processes": n_processes,
        "overwrite": overwrite,
        "normalize_stains": normalize_stains,
        "min_intersection_ratio": min_intersection_ratio
    }
    
    with open(f'preprocessing_{wsi_name_no_extension}.yaml', 'w') as file:
        # default_flow_style=False ensures a block-style YAML (human readable)
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        

def sh_generate(wsi_folder, wsi_name, target_mag):
    wsi_path = wsi_folder+'/'+wsi_name
    wsi_name_no_extension = wsi_name.rsplit(".", 1)[-2]
    output_path = wsi_path +'/preprocessing/'+wsi_name_no_extension+'/'+wsi_name_no_extension
    
    script_content = f"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yao.lu.yl2499@yale.edu
#SBATCH --job-name=GBM_CellViT
#SBATCH --ntasks=1 --cpus-per-gpu=8
#SBATCH --mem-per-cpu=64g
#SBATCH --time=10:00:00

module load miniconda
conda activate cellvit_env
module load OpenSlide/3.4.1-GCCcore-12.2.0-largefiles
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 ./preprocessing/patch_extraction/main_extraction.py --config ./preprocessing_{wsi_name_no_extension}.yaml >{wsi_name_no_extension}_preprocessing.log

python3 ./cell_segmentation/inference/cell_detection.py --model ./models/pretrained/CellViT-SAM-H-x40.pth --gpu 0 --magnification {target_mag} --geojson process_wsi --wsi_path {wsi_path} --patched_slide_path {output_path} >{wsi_name_no_extension}_segmentation.log
    """
    
    with open(f"CellViT_{wsi_name_no_extension}.sh", "w") as f:
        f.write(script_content)

    os.chmod(f"CellViT_{wsi_name_no_extension}.sh", 0o755)
    