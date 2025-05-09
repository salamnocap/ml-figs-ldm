import os, sys
import glob
import argparse
import json
import numpy as np
import torch_fidelity
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ldm.data.ml_figs import MlFigsValidation


class GeneratedSamplesDataset(Dataset):
    def __init__(self, generated_samples_path: str):
        # path to generated samples
        self.samples = glob.glob(os.path.join(generated_samples_path, "*.png"))
        
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
    def __init__(self, mlfigs_dataset):
        self.mlfigs_dataset = mlfigs_dataset

    def __len__(self):
        return len(self.mlfigs_dataset)

    def __getitem__(self, idx):
        sample = self.mlfigs_dataset.__getitem__(idx)
        return transforms.ToTensor()(sample['image']).to(torch.uint8)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--generated_samples_path', 
    type=str, 
    default='outputs/generated_samples_dataset/samples', 
    help='path to generated samples'
)
parser.add_argument(
    '--test_samples_path', 
    type=str, 
    default = 'dataset/ml-scicap-figs/test.json', 
    help='json file to test set'
)
parser.add_argument(
    '--output_file_name', 
    type=str, 
    default = 'metrics', 
    help='json file to test set'
)
args = parser.parse_args()


def main():
    gen_dataset = GeneratedSamplesDataset(args.generated_samples_path)
    test_dataset = MlFigsValidation(
        json_file=args.test_samples_path,
        size=512,
        text_modality=0,
        random_crop=False,
        square_pad=True,
        use_roi_bboxes=True,
        transform=None
    )
    test_dataset = TestDataset(test_dataset)

    metrics = torch_fidelity.calculate_metrics(input1=gen_dataset, input2=test_dataset, fid=True, isc=True, kid=True)

    with open(args.output_file_name, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()