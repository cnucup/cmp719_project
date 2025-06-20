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
```

### 4. Prepare the Datasets

Since the original datasets contain TIF files and high resolution images, in order to make them compatible with MAT architecture we need to convert them to PNG and divide them into patches.

Use mogrify to convert TIF files into PNGs.

```bash
mogrify -path /destination_path_for_png_images -format png *.tif
```
```bash
for f in *.png; do convert "${f}" +repage -crop 512x512 /path_to_png_tiles/${f%.*}_%04d.png; done;
```

Not all final tiles will have a size of 512*512. To remove small tiles copy or move the `rm_small_tiles.sh` from this repo into the `MAT` directory and use:
```bash
bash rm_small_tiles.sh
```

### 5. Change the metrics accordingly

During the evaluation we use different number of image tiles. To do so, we need to update the 'metric_main.py' in the original MAT repo. You can use the 'metric_main.py' shared in this repos directly, or you can add custom metric functions inside this file.

## Run the training code on pre-trained MAT model

Use the following code to fine-tune the MAT model pre-trained on Places dataset (change the number of GPUs and GPU IDs accordingly) : 

```bash
CUDA_VISIBLE_DEVICES=1,2 python3 train.py --outdir=/models --gpus=2 --data=/path_to_training_set --data_val=/path_to_validation_set --batch=8 --lr=0.001 --resume=/models/Places_512_FullData.pkl --metrics=fid_custom_5k --augpipe bgcfn --kimg=120
```

## Run the test code on fine-tuned model

Use the following code to generate images using the fine-tuned model : 

```bash
python3 generate_image.py --network /path_to_model_pkl_file --dpath /path_to_test_set --mpath /path_to_test_masks --outdir /path_to_output
```
