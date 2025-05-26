## BoneCoT: Multi-center Validation of a whole-body skeleton foundation model for bone metastases prediction using oncologist-derived chain of thought

This is the official repo for "BoneCoT: Multi-center Validation of a whole-body skeleton foundation model for bone metastases prediction using oncologist-derived chain of thought". This repository includes code for BoneFM, which uses CT images to pretrain a ViT-14/g model based on DINOv2 methodology, as well as code for the fine-tuning phase, which covers both direct fine-tuning of BoneFM and the BoneCoT fine-tuning approach.


### Repository Structure

This repository is divided into two main parts:

1. **Pre-training**: The `BoneCoT/pretrain` directory contains the official BoneFM pretraining code based on DINOv2. We made simple modifications to the dataset part and implemented the code using PyTorch version > 2.1 with dinov2-patch.

2. **Fine-tuning**: The `BoneCoT/finetune` directory includes code for fine-tuning and testing BoneFM and BoneCoT models. This section contains training scripts, evaluation tools, and configuration files for both direct fine-tuning and the BoneCoT approach.

### 🔧 Install Environment

1. **Create environment with conda:**

    ```sh
    conda create -n bonecot python=3.9 -y
    conda activate bonecot
    ```

2. **Download repo:**

    ```sh
    # Download and unzip from anonymous repository (temporary link before publication)
    wget https://anonymous.4open.science/r/BoneCoT-4DCF/download -O BoneCoT-4DCF.zip
    unzip BoneCoT-4DCF.zip
    cd BoneCoT-4DCF
    ```

    Note: After paper acceptance, the code will be publicly available on GitHub.

3. **Install PyTorch and CUDA:**

    - To install PyTorch via conda:

        ```sh
        conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
        ```

    - To install PyTorch via pip:

        ```sh
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
        ```

4. **Install additional dependencies:**

    ```sh
    pip install -r requirement.txt
    ```

5. **[Optional] Install DINOv2 pre-training environment:**

    If you want to run DINOv2 pre-training, follow these additional steps:

    ```sh
    # Create a new conda environment for DINOv2
    conda env create -f pretrain/conda.yaml
    conda activate dinov2_new

    # Install DINOv2 package
    cd pretrain
    pip install -e .
    ```

    Note: This step is only required if you plan to run DINOv2 pre-training. For using pre-trained models or fine-tuning, the main environment (bonecot) is sufficient.

Following the above step, you should have all the necessary dependencies installed for all three parts of the repository.


### Assets, checkpoints and Data preparation
We introduce how to prepare the data, model weights and assets for pre-training, fine-tuning and inference.

1. **Download model checkpoints:**

    ```sh
    # Download BoneFM base model (.pth file)
    wget "https://drive.google.com/file/d/1NsiBZOx7vAYiN0IDdjYdqFkfArrW_Scn/view?usp=sharing" -O BoneFM.pth

    # Download BoneCoT model weights for primary/metastatic inference
    wget "https://drive.google.com/file/d/1Be9GyeDTnXjxJ6KA8wYUFHDM6fKmnixr/view?usp=sharing" -O bonecot_weights.zip
    ```

2. **Extract BoneCoT model weights:**

    ```sh
    # Create checkpoints directory if it doesn't exist
    mkdir -p finetune/models/checkpoints

    # Extract BoneCoT weights to the correct location
    unzip bonecot_weights.zip -d finetune/models/checkpoints
    ```

The BoneCoT model weights should be placed in the `finetune/models/checkpoints` directory for the inference code to locate them correctly.




### 🌱Play with `BoneCoT_inference.ipynb`
In the notebook `BoneCoT_inference.ipynb`, we provide a minimal example demonstrating how to use BoneCoT for CT image diagnosis. The notebook covers:

1. Loading configuration of BoneCoT
2. Running inference with BoneCoT
3. Interpreting the model's predictions for:
   - Bone lesion detection
   - Benign or malignant 
   - Primary or metastatic
   - Specific tumor characteristics (osteoblastic/osteolytic)
   - Complications (spinal cord compression, pathological fracture, etc.)
   - ...

The notebook provides step-by-step guidance with example code to help you get started with BoneCoT inference.
