# ML-FIGS-LDM

Create a conda environment

```
conda env create -f environment.yaml
conda activate ml-figs-ldm
pip install -e .
```

To train the ldm from scratch, run the following command:

```
python main.py --config configs/ml-figs-ldm.yaml --train=True
```

Update albumentations package 

``` 
python scripts/update_albm_package.py
```
