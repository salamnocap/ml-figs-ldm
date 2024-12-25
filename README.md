# ML-FIGS-LDM

Create a conda environment:
```bash
conda env create -f environment.yaml
conda activate ml-figs-ldm
pip install -e .
```
 or 
```bash
conda env update --file env.yml
pip install -e .
```

Train LDM (Ml-Figs):
```bash
python main.py --config configs/ml-figs-ldm.yaml --train=True
```
or 
```bash
python main.py --base configs/ml-figs-ldm.yaml --train=True
```

Train VAE (Ml-Figs):
```bash
python main.py --config configs/ml-figs-vae.yaml --train=True
```

Train LDM (Ml-Figs + SciCap ACL):
```bash
python main.py --config configs/ml-figs-scicap-ldm.yaml --train=True
```

Train VAE (Ml-Figs + SciCap ACL):
```bash
python main.py --config configs/ml-figs-scicap-vae.yaml --train=True
```

Evaluate VAE:
```bash
python main.py scripts/eval_vae.py
```
