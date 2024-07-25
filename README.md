# PRnet: Predicting transcriptional responses to novel chemical perturbations using deep generative model

This repository hosts the official implementation of PRnet, a flexible and scalable perturbation-conditioned generative model predicting transcriptional responses to novel complex perturbations at bulk and single-cell levels.

<p align="center"><img src="https://github.com/Perturbation-Response-Prediction/PRnet/blob/main/img/PRnet.svg" alt="PRnet" width="900px" /></p>

## Download model and datasets
We provide [model pretrained weight](http://prnet.drai.cn:9003/tcm/download/?file_path=/mnt/data/PRnetWeb/PRnet_model.zip) and precessed dataset ([LINCS_L1000](http://prnet.drai.cn:9003/tcm/download/?file_path=/mnt/data/PRnetWeb/Lincs_L1000.h5ad) and [Sci-Plex](http://prnet.drai.cn:9003/tcm/download/?file_path=/mnt/data/PRnetWeb/Sci_Plex.h5ad)) for training and reproducibility.

To clone our model, install github and run:
```
git clone https://github.com/Perturbation-Response-Prediction/PRnet.git
```
Please download the datasets and store them in the `dataset` folder. Download the pretrained weights and store them in the `checkpoint` folder.

- PRnet/: contains the code for the model, the data, and documents.
- data: contains the `utils` for data  processing.
- dataset: contains datasets.
- figure: contains notebooks for generate figures in our paper.
- img: contains the graphic abstract of PRnet.
- models: contains models of PRnet.
- trainer: contains trainer of PRnet.
- preprocessing: contains notebooks for processing the data.
- train and test: including test_lincs.py, train_lincs.py, train_sciplex.py and test_sciplex.py.

## Step 1: Installation
We recommend using Anaconda or Dockerfile  to creat environment for using PRnet. Please make sure you have installed pre-installation.
#### Setup the environment with Anaconda
We recommend using Anaconda to create a conda environment. You can create a python environment using the following command:

```
conda create -n PRnet python=3.7
```
Then, you can activate the environment using:

```
conda activate PRnet
pip install -r requirements.txt
```
#### Setup the environment with Docker
Alternatively, you can use Docker to set up the environment. Build the Docker image using the provided Dockerfile:
```
docker build -t prnet .
```
You can change the Docker image of PyTorch according to your cuda version in Dockerfile `FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`.

Then run the Docker container:
```
docker run -dit --gpu all -v $(pwd):/workspace --name PRnet prnet
```
After the docker container is started, you can exit using `exit`. If you use the -d mode, docker will be placed in the background. You can enter the container through `docker attach PRnet`. After stopping the container, start the container with `docker start PRnet`. You also can use `docker exec -it PRnet /bin/bash` to enter the container. 

## Step 2: Test with demo datatset
Inference with demo dataset:
```
python test_demo.py --split_key demo_split
                    --data_path ./datasets/demo.h5ad
                    --results_dir  ./results/demo/
```
## Step 3: Inference with custom datatset

Please see [custom_data_preprocessing.ipynb](preprocessing/custom_data_preprocessing.ipynb) to prepare your dataset. 'custom_data_preprocessing' is a demo which preprocesses the data from CCLE.

## Step 4: Train and test with provided datatset
To train the L1000 dataset:
```
python train_lincs.py --split_key drug_split_4
             
```
To test the L1000 dataset:
```
python test_lincs.py --split_key drug_split_4
             
```
To train the Sci-plex dataset:
```
python train_sciplex.py --split_key drug_split_0
             
```
To test the L1000 dataset:
```
python test_sciplex.py --split_key drug_split_0
             
```


## Figures

| Name                                     | Description                                                  |
| ---------------------------------------- | ------------------------------------------------------------ |
| [drug_candidates_recomandation.ipynb](figure/drug_candidates_recomandation.ipynb) | Recomand drug for diseases.                                  |
| [latent_tsne_lung_cancer](figure/latent_tsne_lung_cancer.ipynb)       | Learnable latent space of lung cancer data                   |
| [SCLC_plot_dsea](figure/SCLC_plot_dsea.ipynb)                | Enrichment score of candidates against small cell lung cancer |


## License
This project is covered under the Apache 2.0 License.




