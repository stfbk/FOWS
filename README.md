<!-- 
# TODO
- add repo description (link to paper)
- explain how to reproduce
    - add requirements (conda env file or pyenv?) and how to install them
    - add explaination on how the dataset was preprocessed
    - add explaination on how to train and test the models
- add links to models and dataset
    - add dataset release disclaimer
- add citation reference 
-->

<!--### Description -->
# Spotting tell-tale visual artifacts in face swapping  videos: strengths and pitfalls of CNN detectors
<!-- ARXIV -->   
<!-- https://github.com/zsxoff/arxiv-badge -->
[![arXiv](https://img.shields.io/badge/arXiv-2506.16497-b31b1b.svg)](https://arxiv.org/abs/2506.16497)

This is the official repository of **Spotting tell-tale visual artifacts in face swapping  videos: strengths and pitfalls of CNN detectors**, presented at [**IWBF2025**](https://www.unibw.de/iwbf2025/program/tech_program) and available on [arXiv](link2arxivPaper). -> add link to paper

The trained models are available at the following [oneDrive folder](https://fbk-my.sharepoint.com/:u:/g/personal/rziglio_fbk_eu/EQRaaxFKzIFApj2GwHUot98BL3LuY9rlyiJgJXYFmoQm-Q?e=xeI3Hk).

We made available for **reserach purposes only** our novel FOWS dataset. You can request access to our dataset by filling this [Google form](https://forms.gle/1cpuDCo6FHZcBvwJ8).

<!-- ############################## -->

# Setup

## Install requirements
- install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)
- create the 'fows' environment with python 3.10 
    ```bash
    conda create -n fows python=3.10
    conda activate fows
     ```
- clone the repository and install the requirements
    ```bash
    # clone project (NOTE: update link)
    git clone https://github.com/RickyZi/FOWS_test.git

    # install project   
    cd FOWS_test
    # activate the conda env
    conda activate fows
    # install the requirements
    pip install -r requirements.txt
    ```   

<!-- ############################## -->

# Usage

## Quick run
<!-- A demo demonstrating the pipeline of the work is available in colab. -->

<!-- If you want to test the pre-trained model on the FOWS dataset or on your own videos: -->

A simple demo explainin the whole pipeline of the project is available in colab. You can use this demo to test the pre-trained model on the FOWS dataset or on your own videos.

[FOWS demo notebook](https://github.com/RickyZi/FOWS_test/blob/main/notebook_demo/FOWS_demo.ipynb) <a target="_blank" href="https://drive.google.com/file/d/1HplmCvSokPsQgWg8qvovZYoCk9PakhF0/view?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

## Dataset preprocessing
<!-- NOTE: add link to dataset and form to download the dataset

Explain how the dataset has been preprocessed and where it should be placed. 

The code expects the dataset to be placed under the [data/](https://github.com/RickyZi/FOWS_test/tree/main/dataset) folder. -->

<!-- You need to preprocess the dataset in order to extract the faces from all videos.  -->

Our FOWS dataset consists in a collection of original and manipulate videos of user performing actions that occlude portion of their face. In order to train the models, we extracted the user faces from the video and organize them in 'occluded' and 'non-occluded'. For ease of reproduction, we also made available an already preprocessed version of the FOWS dataset. You can access it by filling out this [Google form](https://forms.gle/1cpuDCo6FHZcBvwJ8).

You can replicate this preprocessing by using the scripts available in the [./preprocessing/](https://github.com/RickyZi/FOWS_test/tree/main/preprocessing) folder:
- [frames_and_faces_extraction.py](https://github.com/RickyZi/FOWS_test/blob/main/preprocessing/frames_and_faces_extraction.py) will apply mediapipe's Blaze Face detector to detect and extract the faces from the video,
- [fows_dataset_processing.py](https://github.com/RickyZi/FOWS_test/blob/main/preprocessing/fows_dataset_processing.py) will organize the images into 'occluded' and 'non-occluded' faces. Please note that a manual revision of the results may be needed in this case.

In our work we applied the same frame categorization preprocessing to the [GOTCHA dataset](https://github.com/mittalgovind/GOTCHA-Deepfakes) using the [./preprocessing/gotcha_dataset_preprocessing.py](https://github.com/RickyZi/FOWS_test/blob/main/preprocessing/gotcha_dataset_preprocessing.py) script to organize occluded and non-occluded faces. 

The same preprocessing applied to our FOWS dataset can be replicated to preprocess your own videos to be tested with our pre-trained models.

<!-- ############################## -->

## Model training
<!-- explain how to train the model and where the results will be saved
explain what are the commands and how to run code -->

The code for training the models presented in the paper is provided in [train.py](https://github.com/RickyZi/FOWS_test/blob/main/train.py).

You can train a specific model using the following command:
 ``` 
    python train.py --model mnetv2 --train_dataset fows_occ --ft --tags mnetv2_fows_occ_FT
``` 
- model: defines the model backbone used for training 
    - MobileNetV2 (mnetv2)
    - EfficientNetB4 (effnetb4)
    - XceptionNet (xception)
- train_dataset: the dataset used for training (fows_occ, fows_no_occ)
- ft (or tl): the model training strategy
    - ft: Fine-Tuning
    - tl: Transfer Learning
- tags: defines the name of the folder where the model weights and the training logs will be saved


<!-- ############################## -->

## Inference
<!-- NOTE: add link to trained models
explain how to perform inference on the trained models (also the baselines), what kind of data are given in output and where are the results saved. -->

The code to perform inference of the trained models on a specific test_dataset is provided in [test.py](https://github.com/RickyZi/FOWS_test/blob/main/test.py).

You can test a trained model on a specific dataset with the following command:
```
python test.py --model mnetv2 --train_dataset fows_occ --test_dataset fows_no_occ --tl 
```
- model: name of the pre-trained model to use
- train_dataset: the dataset used for training the model (fows_occ, fows_no_occ)
- test_dataset:  the dataset used for testing the model (fows_occ, fows_no_occ)
- ft (or tl): the model training strategy 
    - ft: Fine-Tuning
    - tl: Transfer Learning
--tags: the name of the folder where the model inference results and logs will be saved


We also provide the code for computing GradCam activations for a given dataset in the [gradcam.py](https://github.com/RickyZi/FOWS_test/blob/main/gradcam.py) script.
Example usage:
```
    python gradcam.py --model mnetv2 --train_dataset fows_occ --test_dataset fows_no_occ --ft --cam_method gradcam++ --num-layers 1 --tags mnetv2_fows_occ_FT_vs_fows_no_occ
```
- model: name of the pre-trained model to use
- train_dataset: dataset used when training the model
- test_dataset: dataset used for model inference
- ft (or tl): training strategy
    - ft: Fine Tuning
    - tl: Transfer Learning
- cam_method: which gradcam method to apply (gradcam, gradcam++, eigencam, scorecam)
- num_layers (1,2, or 3): how many layers to use for computing the gradcam output. One layer referes to the last convolutional layer of the model. More than one layer and the gradcam activations will be computed as the average of the layers activation.
- tags: the name of the folder where to save the gradcam activations

<!-- ############################## -->

# Citation
If your research uses part of our dataset, models and code, partially or in full, please cite:
``` 
    @misc{ziglio2025fows,
      title={Spotting tell-tale visual artifacts in face swapping videos: strengths and pitfalls of CNN detectors}, 
      author={Riccardo Ziglio and Cecilia Pasquini and Silvio Ranise},
      journal = {arXiv preprint arXiv: 2506.16497},
      year={2025}, 
    }
```

<!-- ############################## -->
