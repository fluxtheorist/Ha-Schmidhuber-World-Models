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

# Flatten the conv output
flattened = after_conv4.view(after_conv4.size(0), -1)
print(f"Flattened shape: {flattened.shape}")

# Linear layer to get to the latent dimensions
fc = nn.Linear(in_features=256 * 4 * 4, out_features=32)
z = fc(flattened)
print(f"Latent z shape: {z.shape}")

# Linear layer to expand z back to 4096
fc_decode = nn.Linear(in_features=32, out_features=256 * 4 * 4)
z_expanded = fc_decode(z)
print(f"Expanded z shape:{z_expanded.shape}")

# Unflatten back to original
unflattened = z_expanded.view(z_expanded.size(0), 256, 4, 4)
print(f"Unflattened shape:{unflattened.shape}")

# Decoder
transpose1 = nn.ConvTranspose2d(
    in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
)
transpose2 = nn.ConvTranspose2d(
    in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
)
transpose3 = nn.ConvTranspose2d(
    in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
)
transpose4 = nn.ConvTranspose2d(
    in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1
)

# Pass first layer
after_transpose1 = transpose1(unflattened)
print(f"After transpose 1:{after_transpose1.shape}")

# Pass second layer
after_transpose2 = transpose2(after_transpose1)
print(f"After transpose 2:{after_transpose2.shape}")

# Pass third slayer
after_transpose3 = transpose3(after_transpose2)
print(f"After transpose 3:{after_transpose3.shape}")

# Pass fourth layer
after_transpose4 = transpose4(after_transpose3)
print(f"After transpose 4:{after_transpose4.shape}")
