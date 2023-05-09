# Environment requirement
We recommend using the conda environment.
```
# basic environment
CUDA 10.2  # Must use this specific version. Please follow https://developer.nvidia.com/cuda-10.2-download-archive. 
python 3.6

# pytorch version
pytorch==1.5.1

# run after installing correct Pytorch package
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install torch-geometric==1.5.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirement.txt
```

# Dataset preparation

## Download
Download from the link in our paper and unzip them. 

## Datapath
5 dataset is used in our paper as follows:
Tongji.NCA: NCA
Tongji.NCM: NCM
Tongji.NCM_NCA: NCM_NCA
Tsinghua: Tshall
MIT: NEall

Please make sure the name of each dataset is not changed (such as ./NCA, ./NCM, ./NCM_NCA, ./Tshall, ./NEall).

To run experiments with each dataset, 
one should manually change the variable
`data_path` in `*.json` files. For example, if you want to use NCA dataset in GCN train, then you
should open `dl_param.json` and change `data_path='/data/log/yangqi/NatureCom/NCA',`. 
(An easy way to do so is to use Ctrl+F to search the name of the variables.)

# Train
Please check the `*.json` files carefully for hyperparameter settings. 
`dl_param.json` is used to train GCN.
`maml_param.json` is used to train maml.
`ft_param.json` is used to finetune.

`window_len` and `train_num` in `*.json` files is used to change the window length and the quantityof train files

## GCN train
To start training GCN, run
```
cd code
python dl_train.py 
```

After training, the reconstruction errors of data are recorded  in `save_model_path` configured by the
`dl_train.json` file.

## maml train
To start training maml, run
```
cd code
python maml_train.py 
```

After training, the reconstruction errors of data are recorded  in `save_model_path` configured by the
`maml_train.json` file.

## finetune train
Please make sure you have run maml train before finetune, and there are basemodel in `./maml_save`.

Ensure the maml basemodel and `ft_train.json` have the same parameters.

To start finetuning, run
```
cd code
python ft_train.py 
```

After training, the reconstruction errors of data are recorded  in `save_model_path` configured by the
`ft_train.json` file.



