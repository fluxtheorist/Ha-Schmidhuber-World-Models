import torch
import torch.nn as nn

# Single conv layer
conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)

# Create fake 64 x 64 RBG
fake_image = torch.randn(1, 3, 64, 64)

# Pass first layer
after_conv1 = conv1(fake_image)

# Pass second layer
after_conv2 = conv2(after_conv1)

# Pass third slayer
after_conv3 = conv3(after_conv2)

# Pass fourth layer
after_conv4 = conv4(after_conv3)

print(f"Input shape: {fake_image.shape}")
print(f"After conv1: {after_conv1.shape}")
print(f"After conv2: {after_conv2.shape}")
print(f"After conv3: {after_conv3.shape}")
print(f"After conv4: {after_conv4.shape}")
