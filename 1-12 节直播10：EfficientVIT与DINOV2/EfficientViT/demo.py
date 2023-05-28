import torch

from classification.model.build import EfficientViT_M4
model = EfficientViT_M4(pretrained='efficientvit_m4')
image =torch.randn(2,3,224,224)
out = model(image)
print(out)