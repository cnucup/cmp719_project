# Satellite Image Inpainting with GANs
Hacettepe University 2025 CMP719 Course Term Project

This project explores the applicability of Generative Adversarial Networks (GANs) for the image inpainting problem in satellite imagery. It is based on the [Mask-Aware Transformer (MAT)](https://github.com/fenglinglwb/MAT) architecture, which was originally designed for natural image inpainting.

In this work, the pre-trained MAT model is fine-tuned using two satellite image datasets: xView and INRIA Aerial Image Labeling. The goal is to evaluate the performance of GAN-based inpainting models on high-resolution remote sensing imagery, where challenges such as large occlusions, diverse land cover, and sharp structural patterns are prominent.

## Key Features
- Fine-tuning of the MAT model on satellite datasets (xView and INRIA).

- Custom preprocessing and masking strategies to simulate occlusions in satellite images.

- Comparative experiments on different training setups and datasets.

- Visualizations of inpainting results across training stages.

## Objective
To demonstrate the potential of GAN-based models, particularly MAT, for realistic and structure-preserving inpainting in satellite images—supporting applications in remote sensing, urban monitoring, and disaster recovery.
## Installation
Follow these steps to set up the environment and prepare the datasets:

### 1. Clone the MAT Repository

```bash
git clone https://github.com/fenglinglwb/MAT.git
cd MAT
```

### 2. Dockerize the Environment

This project uses a custom `Dockerfile` to ensure compatibility and reproducibility. The `Dockerfile` is included in **this repository**.

Copy or move the `Dockerfile_mat` from this repo into the `MAT` directory, then build and run the Docker container:

```bash
# Inside the MAT directory
cp /path/to/this-repo/Dockerfile_mat .

# Build the Docker image
docker build -f Dockerfile_mat -t mat_project .

# Run the container
docker run --ipc=host --runtime=nvidia --privileged --gpus all -d -v /path_to_mat_folder:/workspace -v /path_to_datasets:/dataset -v /path_to_models:/models --name mat_project **container_id**
```

> ⚠️ Make sure you have Docker and NVIDIA Container Toolkit installed for GPU support.

### 3. Download the Datasets

Prepare the datasets required for fine-tuning:

- **xView Dataset**  
  Download from the official [xView website](https://xviewdataset.org) (registration required).

- **INRIA Aerial Image Dataset**  
  Download from [INRIA’s official site](https://project.inria.fr/aerialimagelabeling/).

Once downloaded, organize the datasets into the appropriate directory structure under `./datasets/`, for example:

```
datasets/
├── xview/
│   ├── images/
│   └── masks/
├── inria/
│   ├── images/
│   └── masks/
