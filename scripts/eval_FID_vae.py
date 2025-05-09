import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from ldm.models.autoencoder import AutoencoderKLCustom
from diffusers import AutoencoderKL
from ldm.data.ml_figs import MlFigsValidation
from tqdm import tqdm
from typing import Optional, Union
from torchvision.transforms.functional import to_pil_image


def get_input(batch, k):
    x = [bb[k] for bb in batch]
    x = torch.stack(x, dim=0)
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.to(memory_format=torch.contiguous_format).float()
    return x


def save_images(reconstructions: torch.Tensor, output_dir: str = "outputs", batch: int = 0):
    for j in range(reconstructions.size(0)):
        filename = f"reconstruction_{batch}_{j}.png"
        path = os.path.join(output_dir, filename)
        image = reconstructions[j].clamp(0, 1).cpu()
        pil_img = to_pil_image(image)
        pil_img.save(path)


def save_samples(
    dataloader: DataLoader,
    vae: Optional[Union[AutoencoderKLCustom , AutoencoderKL]],
    custom: bool = False,
    output_dir: str = None
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(dataloader, desc="Saving Reconstructions...")):
        with torch.no_grad():
            images = get_input(batch, 'image').to(device)

            if custom:
                z = vae.encode(images).sample()
                reconstructions = vae.decode(z)
            else:
                z = vae.encode(images).latent_dist.mean
                reconstructions = vae.decode(z).sample

            reconstructions = reconstructions.clamp(0, 1)
            save_images(reconstructions, output_dir, i)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available else "cpu"
    # Define models
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
        ckpt_path="checkpoints/AutoencoderKL(04-12-2024)_epoch_20_dataset_MlFigs.ckpt",
    ).eval().to(device)

    # vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").eval().to(device)

    # Define dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # test_dataset = MlFigsValidation(
    #     json_file="ml-figs/mlfigs_test.json",
    #     # json_file="ml-scicap-figs/test.json",
    #     size=512,
    #     text_modality=0,
    #     random_crop=False,
    #     square_pad=True,
    #     use_roi_bboxes=True,
    #     transform=transform
    # )
    # test_dataloader = DataLoader(
    #     dataset=test_dataset, batch_size=15, num_workers=1, collate_fn=lambda x: x, shuffle=False
    # )

    # Save samples
    # save_samples(test_dataloader, vae, False, output_dir="outputs_sd_(MlFigs)")

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

    # Save samples
    save_samples(test_dataloader, vae, True, output_dir="reconstructed_figures")
