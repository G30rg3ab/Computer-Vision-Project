import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class MidFusionUNET(nn.Module):
    def __init__(self, out_channels=2, features=[64, 128, 256, 512]):
        """
        This model takes two inputs: an image (3 channels) and a guidance map (1 channel),
        processes them separately in the first stage, fuses them, and then continues as UNet.
        """
        super(MidFusionUNET, self).__init__()

        # --- Initial separate processing for the two modalities ---
        # Process image input (3 channels)
        self.img_initial = DoubleConv(in_channels=3, out_channels=features[0])
        # Process guidance input (1 channel)
        self.guidance_initial = nn.Sequential(
            nn.Conv2d(1, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        # Fusion convolution to combine image and guidance features
        # After concatenation, we have features[0]*2 channels. We use a 1x1 conv to reduce it back to features[0].
        self.fuse_conv = nn.Conv2d(features[0]*2, features[0], kernel_size=1)

        # --- UNet Down-sampling path ---
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # We start building downs from the second level in 'features'
        in_channels = features[0]
        for feature in features[1:]:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # --- UNet Up-sampling path ---
        self.ups = nn.ModuleList()
        for feature in reversed(features[1:]):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        # Final 1x1 convolution: now use 128 channels (features[1]) since the last up block outputs 128 channels.
        self.final_conv = nn.Conv2d(features[1], out_channels, kernel_size=1)

    def forward(self, image, guidance):
        # Process the image branch
        img_feat = self.img_initial(image)  # Shape: (N, features[0], H, W)
        # Process the guidance branch
        guidance_feat = self.guidance_initial(guidance)  # Shape: (N, features[0], H, W)
        # Fuse features by concatenation along the channel dimension
        fused = torch.cat([img_feat, guidance_feat], dim=1)  # (N, features[0]*2, H, W)
        # Reduce channel dimension with 1x1 convolution
        x = self.fuse_conv(fused)  # (N, features[0], H, W)

        # Save skip connections AFTER applying the down block, then pool.
        skip_connections = []
        for down in self.downs:
            x = down(x)               # Process the features with the down block
            skip_connections.append(x)  # Save the output as a skip connection
            x = self.pool(x)          # Then pool for the next level
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Up-sampling path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Transposed conv up-sample
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            # Concatenate skip connection
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        
        return self.final_conv(x)

def test():
    # Create dummy inputs: image (3 channels) and guidance (1 channel)
    image = torch.randn((16, 3, 161, 161))
    guidance = torch.randn((16, 1, 161, 161))
    model = MidFusionUNET()
    preds = model(image, guidance)
    print("Predicted shape:", preds.shape)
    # Optionally, assert that output shape matches input spatial dimensions
    assert preds.shape[2:] == image.shape[2:], "Spatial dimensions must match!"

if __name__ == '__main__':
    test()