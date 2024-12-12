# ML-FIGS-LDM

Create a conda environment
```bash
conda env create -f environment.yaml
conda activate ml-figs-ldm
pip install -e .
```

Update albumentations package 
```bash 
python scripts/update_albm_package.py
```

Train LDM:
```bash
python main.py --config configs/ml-figs-ldm.yaml --train=True
```

Train VAE:
```bash
python main.py --config configs/ml-figs-vae.yaml --train=True
```

Evaluate VAE:
```bash
python main.py scripts/eval_vae.py
```
