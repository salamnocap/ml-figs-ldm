import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from ldm.models.autoencoder import AutoencoderKLCustom
from diffusers import AutoencoderKL
from ldm.data.ml_figs import MlFigsValidation

def get_input(batch, k):
    x = [bb[k] for bb in batch]
    x = torch.stack(x, dim=0)
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.to(memory_format=torch.contiguous_format).float()
    return x

def save_samples(dataloader: DataLoader, vae_custom: AutoencoderKLCustom, vae_pretrained: AutoencoderKL, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    custom_output_dir = os.path.join(output_dir, "custom")
    pretrained_output_dir = os.path.join(output_dir, "pretrained")
    os.makedirs(custom_output_dir, exist_ok=True)
    os.makedirs(pretrained_output_dir, exist_ok=True)

    batch = next(iter(dataloader))
    num_samples = min(len(batch), 10)

    for i in range(num_samples):
        data = batch[i]
        original_image = data['image']
        if isinstance(original_image, torch.Tensor):
            original_image = TF.to_pil_image(original_image)

        # Save the original image
        original_image.save(os.path.join(output_dir, f"original_{i}.png"))

        # Process with custom VAE
        with torch.no_grad():
            z_custom = vae_custom.encode(data['image'].unsqueeze(0)).sample()
            reconstructed_custom = vae_custom.decode(z_custom)
            reconstructed_custom = TF.to_pil_image(reconstructed_custom.squeeze().clamp(0, 1))
            reconstructed_custom.save(os.path.join(custom_output_dir, f"custom_reconstruction_{i}.png"))

        # Process with pretrained VAE
        with torch.no_grad():
            z_pretrained = vae_pretrained.encode(data['image'].unsqueeze(0)).latent_dist.mean
            reconstructed_pretrained = vae_pretrained.decode(z_pretrained).sample
            reconstructed_pretrained = TF.to_pil_image(reconstructed_pretrained.squeeze().clamp(0, 1))
            reconstructed_pretrained.save(os.path.join(pretrained_output_dir, f"pretrained_reconstruction_{i}.png"))

    print(f"Saved {num_samples} samples to {output_dir}")


if __name__ == "__main__":
    # Define models
    vae_custom = AutoencoderKLCustom(
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

    vae_pretrained = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").eval()

    # Define dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = MlFigsValidation(
        json_file="ml-figs/mlfigs_test.json", size=512, square_pad=True, use_roi_bboxes=True, transform=transform
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=10, num_workers=1, collate_fn=lambda x: x, shuffle=True
    )

    # Save samples
    save_samples(test_dataloader, vae_custom, vae_pretrained, output_dir="sample_outputs_10")
