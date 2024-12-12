import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
import pandas as pd

from torchmetrics.image import FID, LPIPS
from torchmetrics.functional import mean_squared_error

from ldm.data.ml_figs import MlFigsValidation
from ldm.models.autoencoder import AutoencoderKLCustom
from ldm.modules.losses.textperceptual import TextPerceptualLoss

from diffusers import AutoencoderKL


def evaluate_vae(dataloader: DataLoader, vae: AutoencoderKLCustom, custom: bool=False) -> tuple:
    fid_values, lpips_values, mse_values, text_perceptual_losses = [], [], [], []
    
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            images = get_input(data, 'image')
            bboxes = get_bboxes(data)

            if custom:
                z = vae.encode(images).sample()
                reconstructions = vae.decode(z)
            else:
                z = vae.encode(images).latent_dist.mean
                reconstructions = vae.decode(z).sample

            print(z.size(), reconstructions.size())

            text_perceptual_losses.append(tpl(images, reconstructions, bboxes).item())
            
            lpips_value = lpips(images, reconstructions.clamp(-1, 1))
            lpips_values.append(lpips_value.item())
            
            images_fid = (images * 255).clamp(0, 255).to(torch.uint8)
            reconstructions_fid = (reconstructions * 255).clamp(0, 255).to(torch.uint8)
            
            fid.update(images_fid, real=True)
            fid.update(reconstructions_fid, real=False)
            
            fid_value = fid.compute()
            fid_values.append(fid_value.item())
            fid.reset()
        
            mse_value = mean_squared_error(images, reconstructions)
            mse_values.append(mse_value.item())
        
    return fid_values, lpips_values, mse_values, text_perceptual_losses


def save_metrics_to_dataframe(fid_values, lpips_values, mse_values, text_perceptual_losses, filename="metrics1.csv"):
    data = {
        "FID": fid_values,
        "LPIPS": lpips_values,
        "MSE": mse_values,
        "TEXT_PERCEPTUAL_LOSS": text_perceptual_losses
    }
    
    df = pd.DataFrame(data)
  
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")


def get_input(batch, k):
    x = [bb[k] for bb in batch]
    x = torch.stack(x, dim=0)
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.to(memory_format=torch.contiguous_format).float()
    return x


def get_bboxes(batch):
    return [bb['bboxes'] for bb in batch]


if __name__ == "__main__":
    vae = AutoencoderKLCustom(
        ddconfig={
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        lossconfig={
            "target": "torch.nn.Identity",
        },
        embed_dim=4,
        ckpt_path="/teamspace/studios/this_studio/logs/2024-12-04T10-08-44_ml-figs-vae/checkpoints/last.ckpt",
    ).eval()
    # vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").eval()

    fid = FID(feature=2048)
    lpips = LPIPS(net_type='alex')
    tpl = TextPerceptualLoss().eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = MlFigsValidation(
        json_file="ml-figs/mlfigs_test.json", size=512, square_pad=True, use_roi_bboxes=True, transform=transform
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=4, num_workers=1, collate_fn=lambda x: x
    )

    fid_values, lpips_values, mse_values, text_perceptual_losses = evaluate_vae(test_dataloader, vae)
    save_metrics_to_dataframe(fid_values, lpips_values, mse_values, text_perceptual_losses, filename='metrics_stdiff.csv')