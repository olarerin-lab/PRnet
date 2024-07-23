# PRnet: Predicting transcriptional responses to novel chemical perturbations using deep generative model

This repository hosts the official implementation of PRnet, a flexible and scalable perturbation-conditioned generative model predicting transcriptional responses to novel complex perturbations at bulk and single-cell levels.

<p align="center"><img src="https://github.com/Perturbation-Response-Prediction/PRnet/blob/main/img/PRnet.svg" alt="PRnet" width="900px" /></p>

## Download model and datasets
We provide [model pretrained weight](http://prnet.drai.cn:9003/tcm/download/?file_path=/mnt/data/PRnetWeb/PRnet_model.h5ad) and precessed dataset ([LINCS_L1000](http://prnet.drai.cn:9003/tcm/download/?file_path=/mnt/data/PRnetWeb/Lincs_L1000.h5ad) and [Sci-Plex](http://prnet.drai.cn:9003/tcm/download/?file_path=/mnt/data/PRnetWeb/Sci_Plex.h5ad)) for training and reproducibility.

To clone our model, install github and run:
```
git clone https://github.com/Perturbation-Response-Prediction/PRnet.git
```
Please download the datasets and store them in the `dataset` folder. Download the pretrained weights and store them in the `checkpoint` folder.

- PRnet/: contains the code for the model, the data, and documents.
- data: contains the `utils` for data  processing.
- figure: contains notebooks for generate figures in our paper.
- img: contains the graphic abstract of PRnet.
- models: contains models of PRnet.
- trainer: contains trainer of PRnet.
- preprocessing: contains notebooks for processing the data.
- train and test: including test_lincs.py, train_lincs.py, train_sciplex.py and test_sciplex.py.

## Step 1: Installation
We recommend using Anaconda or Dockerfile  to creat environment for using PRnet. Please make sure you have installed pre-installation.
#### setup the environment with Anaconda
We recommend using Anaconda to create a conda environment. You can create a python environment using the following command:

```
conda create -n PRnet python=3.7
```
Then, you can activate the environment using:

```
conda activate PRnet
pip install -r requirements.txt
```
#### setup the environment with Anaconda
Alternatively, you can use Docker to set up the environment. Build the Docker image using the provided Dockerfile:
```
docker build -t Dockerfile .
```
Then run the Docker container:
```
docker run -it --rm -v $(pwd):/workspace prnet
```
## Step 2: Installation

## Demos

| Name                                     | Description                                                  |
| ---------------------------------------- | ------------------------------------------------------------ |
| [drug_candidates_recomandation.ipynb](demo/drug_candidates_recomandation.ipynb) | Recomand drug for diseases.                                  |
| [latent_tsne_lung_cancer](demo/latent_tsne_lung_cancer.ipynb)       | Learnable latent space of lung cancer data                   |
| [SCLC_plot_dsea](demo/SCLC_plot_dsea.ipynb)                | Enrichment score of candidates against small cell lung cancer |


## License
This project is covered under the Apache 2.0 License.




