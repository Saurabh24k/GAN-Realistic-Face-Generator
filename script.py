import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

class CelebADataset(Dataset):
        def __init__(self, img_dir, partition_file, bbox_file, landmarks_file, attributes_file=None, split='train', transform=None):
            self.img_dir = img_dir
            self.transform = transform
            partition_map = {'train': 0, 'val': 1, 'test': 2}

            partitions = pd.read_csv(partition_file)
            self.img_names = partitions[partitions['partition'] == partition_map[split]]['image_id'].values

            self.bboxes = pd.read_csv(bbox_file, index_col=0)
            self.landmarks = pd.read_csv(landmarks_file, index_col=0)
            self.attributes = pd.read_csv(attributes_file, index_col=0) if attributes_file else None


        def __len__(self):
            return len(self.img_names)


        def __getitem__(self, idx):
            img_name = self.img_names[idx]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path)

            # Crop image based on bounding box
            bbox = self.bboxes.loc[img_name]
            image = image.crop((bbox['x_1'], bbox['y_1'], bbox['x_1'] + bbox['width'], bbox['y_1'] + bbox['height']))

            if self.transform:
                image = self.transform(image)

            # If attributes are used, prepare the attributes vector
            #attributes = None
            if self.attributes is not None:
                attributes = self.attributes.loc[img_name].values.astype('float')
                attributes[attributes == -1] = 0  # Convert from -1,1 to 0,1
                return image, attributes 
            else:
                image

img_dir = '/Users/sau24k/Documents/Projects/CelebA/img_align_celeba/img_align_celeba'
partition_file = '/Users/sau24k/Documents/Projects/CelebA/list_eval_partition.csv'
bbox_file = '/Users/sau24k/Documents/Projects/CelebA/list_bbox_celeba.csv'
landmarks_file = '/Users/sau24k/Documents/Projects/CelebA/list_landmarks_align_celeba.csv'
attributes_file = '//Users/sau24k/Documents/Projects/CelebA/list_attr_celeba.csv'

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = CelebADataset(img_dir, partition_file, bbox_file, landmarks_file, attributes_file, split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


netG = Generator().to(device)
netD = Discriminator().to(device)

# Load previously saved model weights
netG.load_state_dict(torch.load('generator.pth', map_location=device))
netD.load_state_dict(torch.load('discriminator.pth', map_location=device))

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, 100, 1, 1, device=device)
real_label = 1.0
fake_label = 0.0

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Introducing a learning rate scheduler
schedulerD = StepLR(optimizerD, step_size=30, gamma=0.5)
schedulerG = StepLR(optimizerG, step_size=30, gamma=0.5)

num_additional_epochs = 5
start_epoch = 5 

print("Starting Training Loop...")
for epoch in range(start_epoch, start_epoch + num_additional_epochs):
    for i, data in enumerate(train_loader, 0):
        # Check if attributes are included in the batch
        if isinstance(data, list) and len(data) == 2:
            real_images, attributes = data  # Unpack images and attributes
            real_images = real_images.to(device)
            # You can use attributes here if your model uses them
        else:
            real_images = data[0].to(device)  # No attributes are included, just images
        
        b_size = real_images.size(0)
        real_labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        netD.zero_grad()
        # Pass real images through Discriminator
        real_output = netD(real_images).view(-1)
        errD_real = criterion(real_output, real_labels)
        errD_real.backward()
        D_x = real_output.mean().item()

        # Generate fake images and pass through Discriminator
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake_images = netG(noise)
        fake_labels = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        fake_output = netD(fake_images.detach()).view(-1)
        errD_fake = criterion(fake_output, fake_labels)
        errD_fake.backward()
        D_G_z1 = fake_output.mean().item()

        # Update discriminator
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update generator
        netG.zero_grad()
        output = netD(fake_images).view(-1)
        errG = criterion(output, real_labels)  # Trick the generator into thinking fakes are real
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print(f"[{epoch+1}/{start_epoch + num_additional_epochs}][{i}/{len(train_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

    schedulerD.step()
    schedulerG.step()

    # Optionally save generated images at each epoch's end
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    save_image(fake, f'output_epoch_{epoch+1}.png', normalize=True)
    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')

print("Training completed.")
