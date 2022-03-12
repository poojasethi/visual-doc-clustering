# Document Clustering

## Getting started

### Install dependencies in a new conda environment.
`conda env create --name local_env --file=local_env.yml`

### Download datasets
Datasets are available for download [here](https://drive.google.com/drive/folders/1yjovBe7blrTmarF39wk6P_gUwmT0bfk-?usp=sharing).

And should be stored with the following directory structure and names:
`datasets/rvl-cdip/`
`datasets/sroie2019/`

### Download finetuned models.
Models are available for download [here]()

And should be stored with the following directory structure and names:
```
finetuned_models/finetuned_related_lmv1/
finetuned_models/finetuned_unrelated_lmv2/
```

## Training and running models.

## Get document embeddings from LayoutLM (and variants).
`python get_hidden_states.py -r <rivlets_dir> <model_type>`

`python get_hidden_states.py -r datasets/rvl-cdip/rvl_cdip_processed/base/rivlets/ vanilla_lmv1`

### Run unsupervised clustering
`python clustering.py -p datasets/rvl-cdip`
