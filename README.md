[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/salamnocap/ml-figs-ldm)
# ML-FIGS-LDM
## EDUCATIONAL FIGURE GENERATION USING TEXT PERCEPTUAL LOSS

<p align="center">
  <img src="assets/diffusion_process.gif" alt="Diffusion" width="512" />
</p>

ML-FIGS-LDM is a Latent Diffusion Model (LDM) for generating educational figures. The AutoencoderKL is trained using a Text Perceptual Loss to reconstruct more readable text within the figures.

## Dataset [ML-Figs](https://huggingface.co/datasets/salamnocap/ml-figs)
We present the [ML-Figs dataset](https://huggingface.co/datasets/salamnocap/ml-figs), a comprehensive collection of 4,302 figures and captions extracted from 43 machine learning books. This dataset is designed to advance research in understanding and interpreting educational materials. It includes 4000 samples for training and 302 for testing.

## Expanded Dataset [Ml-Figs-SciCap](https://www.kaggle.com/datasets/kuantaiulysalamat/ml-figs-scicap)
To improve the coverage and diversity of our datasets, we decided to expand the ML-Figs dataset by adding extra figures and captions from the [SciCap dataset](https://doi.org/10.48550/arXiv.2110.11624), particularly those from ACL papers. This expansion [ML-Figs + SciCap](https://www.kaggle.com/datasets/kuantaiulysalamat/ml-figs-scicap) has boosted the total size of our dataset to an impressive 19,514 samples.

## Text Perceptual Loss
The Text Perceptual Loss calculates the perceptual similarity between the text regions of two images by extracting text bounding boxes. The mean squared error (MSE) loss is then computed for each corresponding text region. The final loss is the average of these individual region losses.
[Text Perceptual Loss (TPL)](ldm/modules/losses/textperceptual.py)

## Install Dependencies:
```bash
pip install -r requirements.txt
```
or create a conda environment:
```bash
conda env create -f environment.yaml
conda activate ml-figs-ldm
pip install -e .
```

Update albumentations package:
```bash 
python scripts/update_albm_package.py
```

## Train LDM (Ml-Figs + SciCap ACL):
```bash
python main.py --base configs/ml-figs-scicap-ldm.yaml --train=True --scale_lr=False
```

## Train VAE (Ml-Figs + SciCap ACL):
```bash
python main.py --base configs/ml-figs-scicap-vae.yaml --train=True
```

## Evaluate LDM:
```bash
python scripts/eval_ldm.py
python scripts/eval_FID_ldm.py
```

## Evaluate VAE:
```bash
python main.py scripts/eval_vae.py
```

## Qualitative Results:

### Qualitative Comparison of Autoencoder Models:
Model A trained on ML-Figs, Model B trained on ML-Figs + SciCap. TPL: Text Perceptual Loss. SD refers to Stable Diffusion v1-4 trained on LAION.
<p align="center">
  <img src="assets/autoencoder_results.png" alt="Qualitative Comparison" width="600"/>
</p>

### Generated samples across varying classifier-free guidance (CFG) scales:
<p align="center">
  <img src="assets/generated_samples.png" alt="Generated Samples" width="600" />
</p>

## Download Models:
Autoencoder and LDM models are available for download at [huggingface.co/salamnocap/ml-figs-ldm](https://huggingface.co/salamnocap/ml-figs-ldm). The models are trained on the ML-Figs\+SciCap dataset.
