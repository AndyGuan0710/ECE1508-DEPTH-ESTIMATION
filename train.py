import h5py
import numpy as np
import math, tqdm, sys, os

import torch
import torch.nn.modules as nn
import torch.optim as optim
import torch.nn.functional as F

# define the dataset class
class rgbd_dataset(torch.utils.data.Dataset):
    def __init__(self, images, depths):
        super(rgbd_dataset, self).__init__()
        self.images = images
        self.depths = depths
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        image = torch.tensor(self.images[i]) / 255.0
        depth = torch.tensor(np.expand_dims(self.depths[i], axis=0)) / 10.0
        return image, depth


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # encoder
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512,1024)
        # decoder
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)
        # one chnnel for depth estimation
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU()
        )

        # up-sample and down-sample
        self.pool = nn.MaxPool2d(2, 2)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )



    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        # Final depth map
        out = self.final(dec1)
        return out


"""
def supervised_edge_aware_smoothness_loss(predicted, depth_label, image, lambda_edge=10.0):
    
    Computes edge-aware smoothness loss for depth prediction.

    Args:
        predicted (Tensor): Predicted depth map (B, 1, H, W).
        depth_label (Tensor): Ground truth depth map (B, 1, H, W).
        image (Tensor): Input RGB image (B, 3, H, W).
        lambda_edge (float): Weighting factor for edge-aware smoothness.

    Returns:
        Tensor: Loss value.
    
    # Compute depth gradients
    pred_dx = torch.abs(predicted[:, :, :, :-1] - predicted[:, :, :, 1:])
    pred_dy = torch.abs(predicted[:, :, :-1, :] - predicted[:, :, 1:, :])

    label_dx = torch.abs(depth_label[:, :, :, :-1] - depth_label[:, :, :, 1:])
    label_dy = torch.abs(depth_label[:, :, :-1, :] - depth_label[:, :, 1:, :])

    # Compute image gradients (use mean over RGB channels)
    image_gray = torch.mean(image, dim=1, keepdim=True)  # Convert RGB to grayscale
    img_dx = torch.abs(image_gray[:, :, :, :-1] - image_gray[:, :, :, 1:])
    img_dy = torch.abs(image_gray[:, :, :-1, :] - image_gray[:, :, 1:, :])

    # Compute edge-aware weights
    edge_weight_x = torch.exp(-lambda_edge * img_dx)
    edge_weight_y = torch.exp(-lambda_edge * img_dy)

    # Compute smoothness loss
    loss_x = torch.mean(edge_weight_x * torch.abs(pred_dx - label_dx))
    loss_y = torch.mean(edge_weight_y * torch.abs(pred_dy - label_dy))

    return loss_x + loss_y
"""

def compute_gradient(tensor, direction):
    """
    Computes gradients of a tensor along the specified direction.

    Args:
        tensor: Input tensor of shape [B, C, H, W].
        direction: Direction for gradient computation, either 'x' or 'y'.

    Returns:
        Tensor of gradients in the specified direction.
    """
    if direction == 'x':
        return tensor[:, :, :, :-1] - tensor[:, :, :, 1:]
    elif direction == 'y':
        return tensor[:, :, :-1, :] - tensor[:, :, 1:, :]
    else:
        raise ValueError("Invalid direction. Use 'x' or 'y'.")

def supervised_edge_aware_smoothness_loss(predicted, depth_label, lambda_edge=10.0):
    """
    Computes supervised edge-aware smoothness loss using depth labels.

    Args:
        predicted: Predicted depth map of shape [B, 1, H, W].
        depth_label: Ground-truth depth map of shape [B, 1, H, W].
        lambda_edge: Edge sensitivity scaling factor.

    Returns:
        Edge-aware smoothness loss as a scalar.
    """


    # Compute gradients of the depth labels
    label_grad_x = compute_gradient(depth_label, direction='x')
    label_grad_y = compute_gradient(depth_label, direction='y')

    # Compute gradients of the predicted depth
    pred_grad_x = compute_gradient(predicted, direction='x')
    pred_grad_y = compute_gradient(predicted, direction='y')

    # Compute edge-aware weights from depth label gradients
    weight_x = torch.exp(-lambda_edge * torch.abs(label_grad_x))
    weight_y = torch.exp(-lambda_edge * torch.abs(label_grad_y))

    # Compute edge-aware smoothness loss
    loss_x = torch.mean(weight_x * torch.abs(pred_grad_x))
    loss_y = torch.mean(weight_y * torch.abs(pred_grad_y))

    return loss_x + loss_y

def train(model, train_loader, valid_loader, optimizer, device, epochs=10, lambda_edge=0.1):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")

        for i, (images, depths) in tqdm.tqdm(enumerate(train_loader)):
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()

            # Forward pass
            predicted_depths = model(images)
            loss_mse = F.mse_loss(predicted_depths, depths)
            loss_smooth = supervised_edge_aware_smoothness_loss(predicted_depths, depths, lambda_edge)

            # Compute loss with supervised edge-aware smoothness
            loss = loss_mse + 0.1 * loss_smooth

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Training Loss: {train_loss / len(train_loader):.4f}")

        # Validation phase
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, depths in valid_loader:
                images, depths = images.to(device), depths.to(device)
                predicted_depths = model(images)
                loss = supervised_edge_aware_smoothness_loss(predicted_depths, depths, lambda_edge)
                valid_loss += loss.item()

        print(f"Validation Loss: {valid_loss / len(valid_loader):.4f}")

        # Save model
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    # load the dataset from .mat file
    print("loading dataset, waiting...")
    with h5py.File('nyu_depth_v2_labeled.mat', 'r') as f:
        images = f['images'][:]
        depths = f['depths'][:]
    print("image: ", images.shape)
    print("depths: ", depths.shape)

    # create dataset
    # split the whole dataset into two parts: training and validation
    test_num = 200
    train_dataset = rgbd_dataset(images[test_num:], depths[test_num:])
    valid_dataset = rgbd_dataset(images[:test_num], depths[:test_num])
    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False)

    # train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train(model, train_loader, valid_loader, optimizer, device, epochs=30, lambda_edge=0.1)