import torch
import torch.nn as nn
import torchvision.transforms as T
class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=1000):
        super(SwinTransformer, self).__init__()
        # Define the patch embedding layer
        self.patch_embed = nn.Conv2d(in_channels, 96, kernel_size=patch_size, stride=patch_size)
        
        # Define the Swin Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SwinTransformerBlock(dim=96, num_heads=3) for _ in range(12)
        ])
        self.classifier = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])
        x = self.classifier(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(SwinTransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = x + self.ffn(x)
        return x

# Example usage
model = SwinTransformer()
input_image = torch.randn(1, 3, 224, 224)  # Batch of 1 image, 3 channels, 224x224 size
output = model(input_image)
