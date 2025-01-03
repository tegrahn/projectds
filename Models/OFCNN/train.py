# Contributed by Tomas Grahn

import os
import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

#https://github.com/AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow

class FallDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None):
        """
        Args:
            root_dir (string): Directory with subdirectories of frames for each clip.
            sequence_length (int): Number of frames to stack as channels.
            transform (callable, optional): Transform to apply to each frame.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.clips = []  # Each entry: (path_to_clip, label)

        # Scan the directory
        for clip_folder in os.listdir(root_dir):
            label = 1 if "FALL" in clip_folder else 0
            self.clips.append((os.path.join(root_dir, clip_folder), label))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path, label = self.clips[idx]
        frames = []

        # Load each frame in the directory
        for frame_file in sorted(os.listdir(clip_path))[:self.sequence_length]:
            frame_path = os.path.join(clip_path, frame_file)
            frame = Image.open(frame_path)
            frame = self.transform(frame)
            frames.append(frame)

        # Stack frames along the channel dimension
        stacked_frames = torch.cat(frames, dim=0)
        return stacked_frames, label

import torchvision.models as models
import torch.nn as nn

#Multi Channel VGG with last 2 FC layers unfrozen for training and 0.9, 0.8 dropout applied to last 2 fc layers

class VGG16MultiChannel(nn.Module):
    def __init__(self, input_channels, num_classes=1):
        super(VGG16MultiChannel, self).__init__()
        
        # Load the pretrained VGG16 model
        self.vgg = models.vgg16(pretrained=True)
        
        # Modify the input layer to accept the specified number of channels
        self.vgg.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.classifier[2] = nn.Dropout(0.9)  # Adjust dropout based on paper
        self.vgg.classifier[5] = nn.Dropout(0.8) # Adjust dropout based on paper
        # Modify the classifier to output a single class for binary classification
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        self.sigmoid = nn.Sigmoid()
        
        # Freezing all but last 2 layers based on paper
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze all parameters
        # Unfreeze the last two fully connected layers for training
        for param in self.vgg.classifier[3:].parameters():
            param.requires_grad = True  # Unfreeze last 2 layers

    def forward(self, x):
        x = self.vgg(x)
        x = self.sigmoid(x)
        return x

input_channels = (2*24) # 1 second at 25 fps for x and y flow
# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = FallDataset(root_dir="1.0", sequence_length=input_channels, transform=transform)

# Calculate the size of the validation set
val_split = 0.2
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size

# Split the dataset into training and validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Check CUDA
print(torch.cuda.get_device_name(torch.cuda.current_device())) 
device = torch.device('cuda:1')  # Select the second GPU

# Parameters
batch_size = 64
num_epochs = 3000
learning_rate = 0.0001 # 10^-4 
patience = 100 

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Early stopping variables
best_val_loss = float('inf')
epochs_without_improvement = 0

model = VGG16MultiChannel(input_channels = input_channels).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    # Training phase
    model.train()
    for frames, labels in train_loader:
        frames, labels = frames.to(device), labels.to(device).float()
        # Forward pass
        optimizer.zero_grad()
        outputs = model(frames)
        outputs = outputs.squeeze()  # Remove extra dimension for binary classification
        loss = criterion(outputs, labels)
        #print(labels)
        #print(outputs)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # Calculate training loss
    avg_train_loss = running_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device).float()
            outputs = model(frames)
            outputs = outputs.squeeze()  # Remove extra dimension for binary classification
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    # Log training and validation progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_duration}")
    
    # Check if the validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_epoch = epoch
        # Save the model weights with the best validation loss
        torch.save(model.state_dict(),'best_model.pth')
        print(f"Validation loss improved. Saving model at epoch {epoch+1}.")
    else:
        epochs_without_improvement += 1
        print(f"Validation loss did not improve. {epochs_without_improvement}/{patience} epochs without improvement.")
    
    # Early stopping check
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

