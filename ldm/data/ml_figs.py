import json
import PIL
import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from torchvision import transforms

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def visualize_ocr(
    dataloader: DataLoader,
    rows: int = 3, 
    cols: int = 3, 
    figsize: tuple = (15, 15), 
    max_caption_length: int = 40,
    ocr_facecolor: bool = False,
    ocr_edgecolor: bool = False,
    edgecolor: str = "gray",
    linewidth: float = 2.0
):
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, facecolor='white')
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.flatten()

    batch = next(iter(dataloader))
    num_samples = min(len(batch), len(axs))

    for i in range(num_samples):
        ax = axs[i]
        data = batch[i]

        image = data['image']
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        ax.imshow(image, cmap="gray")

        for l, t, w, h in data['bboxes']:
            rect = patches.Rectangle(
                (l, t), w, h, 
                linewidth=1, 
                edgecolor='r' if ocr_edgecolor else 'none', 
                 facecolor='blue' if ocr_facecolor else 'none'
            )
            ax.add_patch(rect)

        caption = data['caption']
        ax.set_title(
            caption[:max_caption_length] + '...' if len(caption) > max_caption_length else caption, 
            fontsize=12, color="black"
        )
        ax.axis("off")

        rect = patches.Rectangle(
            (0, 0), 1, 1, 
            transform=ax.transAxes,
            linewidth=linewidth, 
            edgecolor=edgecolor, 
            facecolor='none'
        )
        ax.add_patch(rect)

    for ax in axs[num_samples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


class MlFigs(Dataset):
    def __init__(
        self, 
        json_file: str, size: int = 224, random_crop=False, 
        square_pad=False, use_roi_bboxes=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    ):
        self.json_file = json_file 
        self.size = size
        self.transform = transform
        self.random_crop = random_crop
        self.square_pad = square_pad
        self.use_roi_bboxes = use_roi_bboxes
        self.data = self._load_data()
        
        if self.square_pad:
            self.square_pad_transform = A.Compose([ 
                A.LongestMaxSize(max_size = self.size),
                A.PadIfNeeded(
                    min_width=self.size, 
                    min_height=self.size, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value = [255, 255, 255]
                )
            ], bbox_params=A.BboxParams(
                format='coco', min_visibility=0.1, label_fields = ['category_ids'], clip=True
            ))
        else:
            self.image_rescaler = A.SmallestMaxSize(
                max_size=self.size, interpolation=cv2.INTER_AREA
            )
            if self.random_crop:
                self.cropper = A.RandomCrop(height=self.size, width=self.size)
            else:
                self.cropper = A.CenterCrop(height=self.size, width=self.size)

            if self.use_roi_bboxes:
                self.bbox_transform = A.Compose([
                    self.image_rescaler,
                    self.cropper
                ], bbox_params=A.BboxParams(
                    format='coco', min_visibility=0.1, label_fields = ['category_ids'], clip=True
                ))

    def _load_data(self):
        with open(self.json_file) as f:    
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def _get_bboxes_tensor(self, ocr: dict):
        return [
            [l, t, w, h] 
            for l, t, w, h in zip(ocr['left'], ocr['top'], ocr['width'], ocr['height'])
        ]

    def sequential_sample(self, idx):
        if idx >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(idx + 1)

    def skip_sample(self, idx):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(idx=idx)

    def __getitem__(self, idx):
        sample = {}
        figure_metadata = self.data[idx]
        ocr_data = figure_metadata['ocr']

        image_path = str(
            Path(self.json_file).parent.parent / figure_metadata['renderURL']
        )

        if self.use_roi_bboxes:
            bboxes = self._get_bboxes_tensor(ocr_data)
            ids = [1 for i in range(len(bboxes))]

        try:
            image = Image.open(image_path)
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            image = np.array(image).astype(np.uint8)

            if self.square_pad:
                tr_im = self.square_pad_transform(
                    image=image, 
                    bboxes=bboxes if self.use_roi_bboxes else [], 
                    category_ids = ids if self.use_roi_bboxes else []
                )
                image = tr_im['image']
                sample['bboxes'] = tr_im['bboxes']
            else:
                image = self.image_rescaler(image=image)['image']
                image = self.cropper(image=image)['image']

            if self.transform:
                image = self.transform(Image.fromarray(image))
            
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_path}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        sample['image'] = image
        sample['caption'] = figure_metadata.get('caption', '')

        return sample
    
    def plot_ocr_features(self, idx: int):
        sample = self.__getitem__(idx)
        if sample is None:
            return
        
        image = sample['image']
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        for l, t, w, h in sample['bboxes']:
            rect = patches.Rectangle(
                (l, t), w, h, linewidth=1, edgecolor='r', facecolor='blue'
            )
            ax.add_patch(rect)

        print(sample['caption'])
        plt.show()

    def plot_figure(self, idx: int):
        sample = self.__getitem__(idx)
        if sample is None:
            return
        
        image = sample['image']
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        print(sample['caption'])
        plt.show()


class MlFigsTrain(MlFigs):
    def __init__(self, **kwargs):
        self.shuffle = True
        super().__init__(**kwargs)


class MlFigsValidation(MlFigs):
    def __init__(self, **kwargs):
        self.shuffle = False
        super().__init__(**kwargs)
