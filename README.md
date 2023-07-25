# SenseXAMP

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Supported Tasks
* **Task1**: AMPs Binary classification.
* **Task2**: AMPs regression.
* **Other task**: Such as AMPs ranking and Multilabel classification, will be published in subsequent papers, commenting soon.

## Installation

```bash
# clone project
git clone https://github.com/William-Zhanng/SenseXAMP.git
cd SenseXAMP

# create conda virtual environment
conda create -n torch1.7 python=3.8 
conda activate torch1.7

# install all requirements
pip install -r requirements.txt
```

## Usage

### Step1: Download our experiments dataset
Please download the original datasets first (the dataset here only contains sequences), and then generate Protein descriptors and ESM-1b embeddings based on these sequences.

**Download here:** https://drive.google.com/drive/folders/1L0OKKq3yQmKQTyFSQ3YUmB5RRnba10w1?usp=sharing

### Step2: Download our model checkpoints to quickly reproduce our results
We are organizing the information and will provide it within two weeks after the article is accepted.

### Step3: Generate protein descriptors and esm-1b embeddings using our scripts

### Step4: Use our codebase for quick experimentation
Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of AMPsMultiModalBenchmark.

## Project structure
### Overview of project structure
- SenseXAMP
  - **Ampmm_base**
    - data
    - models
    - runner
    - utils
  - **configs**
  - **datasets**
  - **tools**
    - cd_hit
    - esm_project
    - strcture_data_generation
    - xxx.py
  - **utils**
    - xxx.py
  - **experiments**
  - **train.py**: 

### 1. Ampmm_base
The implementation of the core class, "Trainer," in our code repository.

### 2. configs
**configs**

- cls_task
  - xxx.py
- reg_task
  - xxx.py
  
This directory stores various configuration files that play a crucial role in managing our codebase. It encompasses model parameters, dataset settings, and hyperparameter configurations, all conveniently organized within config files. 

The naming convention for these config files adheres to the following structure: `datasetname_modelname.py`. For instance, the file `benchmark_imbalanced_fusion.py` signifies the SenseXAMP model's application on our classification imbalanced dataset.

### 3. datasets
This directory serves as a repository for various forms of datasets.
- **datasets**
  - ori_datasets
  - stc_datasets
  - stc_info
  - esm_embeddings