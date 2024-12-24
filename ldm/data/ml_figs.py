import os
import json
import PIL
import cv2
import torch
import pytesseract

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A

from pytesseract import Output
from torchvision import transforms
from string import punctuation

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def visualize_ocr(
    dataloader: DataLoader,
    rows: int = 3, 
    cols: int = 3, 
    figsize: tuple = (15, 15), 
    max_caption_length: int = 40,
    facecolor: bool = False
):
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
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
                (l, t), w - l, h - t, 
                linewidth=1, edgecolor='r', 
                facecolor='blue' if facecolor else 'none'
            )
            ax.add_patch(rect)
        ax.axis("off")

        caption = data['caption']
        ax.set_title(
            caption[:max_caption_length] + '...' if len(caption) > max_caption_length else caption, 
            fontsize=12, color="black"
        )

    for ax in axs[num_samples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


class MlFigs(Dataset):
    def __init__(
        self, 
        json_file: str, size: int = 224, transform=None, 
        random_crop=False, square_pad=False, use_roi_bboxes=False
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

    def _preprocessing(self, text: str) -> str:
        text = text.lower()
        text = ''.join([c for c in text if c not in punctuation])
        return text

    def _get_ocr_results(self, path: str) -> dict:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ocr_result = pytesseract.image_to_data(img, output_type=Output.DICT)
        
        result = {
            'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []
        }
        
        for i in range(len(ocr_result['level'])):
            text = ocr_result['text'][i].strip()
            conf = int(ocr_result['conf'][i])
    
            if conf <= 20 or text == '':
                continue

            is_duplicate = False
            for j in range(len(result['text'])):
                if (result['text'][j] == text and
                    abs(result['left'][j] - ocr_result['left'][i]) < 10 and
                    abs(result['top'][j] - ocr_result['top'][i]) < 10):
                    is_duplicate = True
                    break
            
            text = self._preprocessing(text)
    
            if text == '':
                continue
    
            if not is_duplicate:
                result['text'].append(text)
                result['left'].append(ocr_result['left'][i])
                result['top'].append(ocr_result['top'][i])
                result['width'].append(ocr_result['width'][i])
                result['height'].append(ocr_result['height'][i])
                result['conf'].append(conf)
        
        return result

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

        image_path = str(
            Path(self.json_file).parent.parent / figure_metadata['renderURL']
        )
        
        if 'ocr' in figure_metadata:
            ocr_data = figure_metadata['ocr']
        else:
            ocr_data = self._get_ocr_results(image_path)

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
