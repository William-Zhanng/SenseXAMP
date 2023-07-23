# AMPsMultiModalBenchmark

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

## Step1: Download our experiments datasets
Please download the original datasets first (the dataset here only contains sequences), and then generate Protein descriptors and ESM-1b embeddings based on these sequences.

**Download here:** https://drive.google.com/drive/folders/1L0OKKq3yQmKQTyFSQ3YUmB5RRnba10w1?usp=sharing

## Step2: Download our model checkpoints to quickly reproduce our results
We are organizing the information and will provide it within two weeks after the article is accepted.

## Step3: Generate protein descriptors and esm-1b embeddings using our scripts

## Step4: Use our codebase for quick experimentation
Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of AMPsMultiModalBenchmark.