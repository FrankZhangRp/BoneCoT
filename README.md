## 	BoneCoT: Multi-center validation of a whole-body skeleton foundation model for bone metastases guided by clinician-derived chain of thought

This is the official repo for "BoneCoT: Multi-center validation of a whole-body skeleton foundation model for bone metastases guided by clinician-derived chain of thought". This repository includes code for BoneFM, which uses CT images to pretrain a ViT-14/g model based on DINOv2 methodology, as well as code for the fine-tuning phase, which covers both direct fine-tuning of BoneFM and the BoneCoT fine-tuning approach.


### Repository Structure

This repository is divided into two main parts:

1. **Pre-training**: The `BoneCoT/pretrain` directory contains the official BoneFM pretraining code based on DINOv2. We made simple modifications to the dataset part and implemented the code using PyTorch version > 2.1 with dinov2-patch.

2. **Fine-tuning**: The `BoneCoT/finetune` directory includes code for fine-tuning and testing BoneFM and BoneCoT models. This section contains training scripts, evaluation tools, and configuration files for both direct fine-tuning and the BoneCoT approach.

### 🔧 Install Environment

0. **System Requirements：**


    Before you begin, please ensure your environment meets the following requirements:

    * **OS**: Ubuntu 22.04 (tested on Ubuntu 22.04.4 LTS)
    * **GPU**: NVIDIA GPU with **≥24 GB** VRAM (tested on NVIDIA A100-80G)
    * **NVIDIA Driver**: `>=550.54.15`
    * **CUDA Toolkit**: `12.4`
    * **Python**: 3.9
    * **Conda**: Installed via [Anaconda](https://www.anaconda.com/products/distribution)

    Check with:

    ```sh
    # Check Ubuntu version
    lsb_release -a

    # Check GPU and VRAM
    nvidia-smi

    # Check CUDA version
    nvcc --version

    # Check Python version
    python --version
    ```

1. **Create environment with conda:**

    ```sh
    conda create -n bonecot python=3.9 -y
    conda activate bonecot
    ```

2. **Download repo:**

    ```sh
    # Download and unzip from anonymous repository (temporary link before publication)
    mkdir BoneCoT
    cd BoneCoT
    wget https://anonymous.4open.science/api/repo/BoneCoT-4DCF/zip -O BoneCoT.zip
    unzip BoneCoT.zip
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
    pip install -r requirements.txt
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
We introduce how to prepare the data, model weights and assets for fine-tuning and inference.

1. **Download model checkpoints and datasets:**

    ```sh
    # Install gdown for downloading from Google Drive
    pip install gdown
    
    # Download BoneFM base model (.pth file)
    gdown "1NsiBZOx7vAYiN0IDdjYdqFkfArrW_Scn" -O BoneFM.pth

    # Download BoneCoT model weights
    gdown "1nauKOqltTr121zCqeM0-xSxxf7uIM3Vj" -O bonecot_weights.zip

    # Download Inference data (split archive files)
    gdown "1IuGLOwpuTekO8PyI0aj-Y19onYIe_WjR" -O datasets.zip
    gdown "127_tfzelU-cAroytq4z3_8_qh_0ZlWHC" -O datasets.z01
    gdown "1dutPZw934XhdPYJtoxKv_2oH3f4LxoIm" -O datasets.z02
    gdown "1Sj3UkfDB8tTzjD9F_D30IYc9qVJeA6mP" -O datasets.z03
    gdown "1l_VMPjtiOTyBLZICGHFlpygg_5gr-T2N" -O datasets.z04
    gdown "1K9JgOC0IhjKSWYoNqr6w589OX_t7sExk" -O datasets.z05
    gdown "1ALd5yBrjvgdUUkkR_2EwOr9Yd8nNoS6r" -O datasets.z06
    gdown "158jSc-xEB2VK2tFCM4NyS5oWZ2DM-z-C" -O datasets.z07
    gdown "16lzZSCa4fSlH6LSCxb72spRtkMA1FtSN" -O datasets.z08
    gdown "16lzZSCa4fSlH6LSCxb72spRtkMA1FtSN" -O datasets.z09
    gdown "1ZptePUAyh1qs4oy4HZ7K6XnAggb5HMQn" -O datasets.z10
    gdown "1jdmI29y8lIl6POCtBueihUGTCdGlfPMa" -O datasets.z11
    ```

    **Alternative Download Method**: If the command-line download fails, you can manually download all required files from this shared Google Drive folder: https://drive.google.com/drive/folders/1bg0sD-Q3XttOQ586Y345mf_FOABuroP1?usp=sharing

2. **Create checkpoints directory:**

    ```sh
    # Create checkpoints directory if it doesn't exist
    mkdir -p finetune/checkpoints
    cp BoneFM.pth finetune/checkpoints/
    ```

3. **Extract BoneCoT model weights:**

    ```sh
    # Check if zip/unzip utilities are installed
    if ! command -v zip &> /dev/null || ! command -v unzip &> /dev/null; then
        echo "zip and unzip utilities are required. Installing..."
        # For Ubuntu/Debian
        sudo apt-get update && sudo apt-get install -y zip unzip
        # For CentOS/RHEL
        # sudo yum install -y zip unzip
    fi
    
    # Extract BoneCoT weights to the correct location
    unzip bonecot_weights.zip -d finetune
    ```

4. **Extract inference datasets:**

    ```sh
    # Create data directory if it doesn't exist
    mkdir -p finetune/data
    
    # Combine split archive files and extract to data
    zip -F datasets.zip --out datasets_combined.zip
    unzip datasets_combined.zip -d finetune/data
    ```

The BoneCoT model weights should be placed in the `finetune/checkpoints` directory for the inference code to locate them correctly.




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

### 🔬 5-Fold Cross-Validation Inference Notebooks
We provide three Jupyter notebooks for 5-fold cross-validation inference corresponding to the three main diagnostic tasks:

- `Task1_bone_lesion_inference.ipynb` - Bone lesion detection inference
- `Task2_benign_or_malignant_inference.ipynb` - Benign or malignant classification inference  
- `Task3_primary_or_metastatic_inference.ipynb` - Primary or metastatic classification inference

Each notebook runs inference across all 5 folds of the cross-validation setup, providing comprehensive evaluation results.

### 🚀 Running Jupyter Notebooks

To start using the notebooks, follow these steps:

1. **Navigate to the BoneCoT directory:**
   ```sh
   cd /path/to/BoneCoT
   ```

2. **Activate the BoneCoT conda environment:**
   ```sh
   conda activate bonecot
   ```

3. **Start Jupyter notebook server (recommended to use tmux to prevent accidental closure):**
   ```sh
   # Start a new tmux session
   tmux new-session -d -s bonecot_jupyter
   
   # Attach to the tmux session
   tmux attach-session -t bonecot_jupyter
   
   # Navigate to the BoneCoT directory:
   cd /path/to/BoneCoT

   # Activate the BoneCoT conda environment
   conda activate bonecot

   # Start Jupyter notebook server in the tmux session
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
   
   # After starting the server, you will see a URL with a token in the terminal output
   # Copy the token from the output, then:
   # 1. Open your browser and navigate to http://your_server_ip:8888
   # 2. Enter the token when prompted
   # 3. Set a password for future logins
   # 4. You can now access the Jupyter notebook interface
   ```
   
   **Note:** Using tmux is recommended to keep the Jupyter notebook server running even if your terminal session is accidentally closed. You can detach from the tmux session using `Ctrl+B` followed by `D`, and reattach later using `tmux attach-session -t bonecot_jupyter`.

4. **Open and run the desired notebook:**
   - Navigate to the notebook files in your browser
   - Select the task-specific notebook you want to run
   - Execute the cells to perform inference
