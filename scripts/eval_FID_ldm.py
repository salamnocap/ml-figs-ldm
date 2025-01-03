import os
import glob
import argparse
import numpy as np
import torch_fidelity
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


class GeneratedSamplesDataset(Dataset):
    def __init__(self, path_model, cfg):
        # path to generated samples
        self.generated_samples_path = os.path.join(path_model, f'evaluation_cfg_{cfg}')
        self.samples = []
        for _, dirs, _ in os.walk(self.generated_samples_path):
            samples = []
            for dir in dirs:
                samples = glob.glob(os.path.join(os.path.join(self.generated_samples_path, dir), "0*.png"))
                self.samples += samples
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # read image and return tensor
        image = Image.open(sample)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)
        return transforms.ToTensor()(image).to(torch.uint8)

class TestDataset(Dataset):
    def __init__(self, paper2fig_dataset):
        self.paper2fig_dataset = paper2fig_dataset

    def __len__(self):
        return len(self.paper2fig_dataset)

    def __getitem__(self, idx):
        sample = self.paper2fig_dataset.__getitem__(idx)
        return transforms.ToTensor()(sample['image']).to(torch.uint8)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='logs/2024-12-05T08-07-01_ml-figs-ldm/checkpoints/last.ckpt', help='path to model checkpoint')
parser.add_argument('--cfg', type=int, default=10.0, help='cfg scale')
parser.add_argument('--test_samples_path', type=str, default = 'ml-figs/mlfigs_test.json', help='path dir with test samples')
args = parser.parse_args()


def main():
    config_path = glob.glob(os.path.join(args.model_path, "configs/*project.yaml"))[0]
    config = OmegaConf.load(config_path)

    gen_dataset = GeneratedSamplesDataset(args.model_path, args.cfg)
    test_dataset = TestDataset(instantiate_from_config(config.data.params.test))

    metrics = torch_fidelity.calculate_metrics(input1=gen_dataset, input2=test_dataset, fid=True, isc=True, kid=True)
    print(metrics)
    print("done!")

if __name__ == "__main__":
    main()