# 🎭 CelebA GAN - Face Image Generation

This project trains a **Generative Adversarial Network (GAN)** using the **CelebA dataset** to generate realistic face images. The implementation includes a **custom dataset loader** with facial attribute processing, **bounding box-based image cropping**, and a **DCGAN architecture** for image generation.

## 📌 Features
- **Custom Dataset Loader**: Loads CelebA images, bounding boxes, and attributes for training.
- **Data Preprocessing**: Crops images based on bounding box information and applies transformations.
- **Deep Convolutional GAN (DCGAN)**: Uses a Generator and Discriminator for training realistic face images.
- **Training Pipeline**: Supports checkpoint saving, learning rate scheduling, and progress logging.

## 📂 Dataset
- **Images**: Aligned & cropped celebrity faces.
- **Metadata Files**:
  - `list_eval_partition.csv`: Defines training, validation, and test splits.
  - `list_bbox_celeba.csv`: Bounding boxes for facial cropping.
  - `list_landmarks_align_celeba.csv`: Facial landmark data.
  - `list_attr_celeba.csv`: Binary facial attributes (e.g., smiling, glasses).

## 🏗 Architecture
- **Generator**: Uses transposed convolutions and batch normalization.
- **Discriminator**: Uses convolutional layers with LeakyReLU activation.
- **Optimizer**: Adam with **learning rate scheduling**.

## 📊 Training Progress
- Loss values for both Generator and Discriminator are logged.
- Generated images are saved every epoch.
- Final models (`generator.pth`, `discriminator.pth`) are stored.

## 💡 Future Improvements
- Implement conditional GANs for attribute-specific image generation.
- Enhance resolution using **progressive growing GANs (PGGANs)**.
- Experiment with **Wasserstein GANs (WGANs)** for better stability.

## 📩 Contact
For any inquiries, feel free to reach out: **[saurabhrajput24k@gmail.com](mailto:saurabhrajput24k@gmail.com)**

