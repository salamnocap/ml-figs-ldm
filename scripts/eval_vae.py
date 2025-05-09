import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
import pandas as pd

from torchmetrics.image import FID, LPIPS, PSNR, SSIM
from torchmetrics.functional import mean_squared_error

from ldm.data.ml_figs import MlFigsValidation
from ldm.models.autoencoder import AutoencoderKLCustom
from ldm.modules.losses.textperceptual import TextPerceptualLoss

from diffusers import AutoencoderKL


def evaluate_vae(dataloader: DataLoader, vae: AutoencoderKLCustom, custom: bool=False) -> tuple:
    lpips_values, mse_values, text_perceptual_losses, psnr_values, ssim_values = [], [], [], [], []
    
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            images = get_input(data, 'image')
            bboxes = get_bboxes(data)

            if custom:
                z = vae.encode(images.to(device)).sample()
                reconstructions = vae.decode(z)
            else:
                z = vae.encode(images.to(device)).latent_dist.mean
                reconstructions = vae.decode(z).sample

            print(z.size(), reconstructions.size())

            text_perceptual_losses.append(tpl(images, reconstructions.cpu(), bboxes).item())
            
            lpips_value = lpips(images, reconstructions.clamp(-1, 1).cpu())
            lpips_values.append(lpips_value.item())
            
            mse_value = mean_squared_error(images, reconstructions.cpu())
            mse_values.append(mse_value.item())

            psnr_value = psnr(images, reconstructions.cpu())
            psnr_values.append(psnr_value.item())

            ssim_value = ssim(images, reconstructions.cpu())
            ssim_values.append(ssim_value.item())
        
    return lpips_values, mse_values, text_perceptual_losses, psnr_values, ssim_values


def save_metrics_to_dataframe(
    lpips_values, mse_values, text_perceptual_losses, psnr_values, ssim_values, filename="metrics1.csv"
):
    data = {
        "PSNR": psnr_values,
        "SSIM": ssim_values,
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
    device = "cuda" if torch.cuda.is_available else "cpu"
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
        ckpt_path="/teamspace/studios/this_studio/checkpoints/AutoencoderKL(14-02-2024)_epoch_11_dataset_MlFigsSciCap.ckpt"
    ).eval().to(device)
    # vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").eval().to(device)

    lpips = LPIPS(net_type='alex').eval()
    tpl = TextPerceptualLoss().eval()
    psnr = PSNR()
    ssim = SSIM()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = MlFigsValidation(
        # json_file="ml-figs/mlfigs_test.json",
        json_file="ml-scicap-figs/test.json",
        size=512,
        text_modality=0,
        random_crop=False,
        square_pad=True,
        use_roi_bboxes=True,
        transform=transform
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=15, num_workers=1, collate_fn=lambda x: x, shuffle=False
    )

    lpips_values, mse_values, text_perceptual_losses, psnr_values, ssim_values = evaluate_vae(
        test_dataloader, vae, True
    )
    save_metrics_to_dataframe(
        lpips_values, 
        mse_values, 
        text_perceptual_losses,
        psnr_values, 
        ssim_values, 
        filename='metrics.csv'
    )