import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

class TextPerceptualLoss(nn.Module):
    def __init__(self):
        super(TextPerceptualLoss, self).__init__()

    def calculate_perceptual_loss(self, original_image, reconstructed_image, bboxes):
        distances = []

        for bbox in bboxes:
            if len(bbox) < 4:
                continue

            left, top, width, height = map(int, bbox)
            
            original_region = tf.crop(original_image, top, left, height, width)
            reconstructed_region = tf.crop(reconstructed_image, top, left, height, width)
            
            region_distance = F.mse_loss(
                original_region, 
                reconstructed_region, 
                reduction='mean'
            )
            distances.append(region_distance)

        if len(distances) == 0:
            return torch.tensor(0.0, device=original_image.device)
        
        return torch.stack(distances).mean()

    def forward(self, original_image, reconstructed_image, bboxes):
        batch_size = original_image.size(0)
        distances = []

        for i in range(batch_size):
            distances.append(
                self.calculate_perceptual_loss(
                    original_image[i], 
                    reconstructed_image[i], 
                    bboxes[i]
                )
            )
        
        if len(distances) == 0:
            return torch.tensor(0.0, device=original_image.device)
        
        return torch.stack(distances).mean()
