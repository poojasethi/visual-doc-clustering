# Document Clustering

## Getting started

### Install dependencies in a new conda environment.
`conda env create --name local_env --file=local_env.yml`

### Download datasets
Datasets are available for download [here](https://drive.google.com/drive/folders/1yjovBe7blrTmarF39wk6P_gUwmT0bfk-?usp=sharing).

And should be stored with the following directory structure and names:
`datasets/rvl-cdip/`
`datasets/sroie2019/`

## Training and running models.

### Run unsupervised clustering
`python clustering.py -p datasets/rvl-cdip`
