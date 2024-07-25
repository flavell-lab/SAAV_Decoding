# SAAV Decoding

This repository contains all the elements of decoding SAAV's post reversal turn signal.

[DecodingMain.ipynb](https://github.com/flavell-lab/SAAV_Decoding/blob/main/DecodingMain.ipynb): a Jupyter Notebook that can be run through to completion to recreate all results.

[kfold.py](https://github.com/flavell-lab/SAAV_Decoding/blob/main/kfold.py): contains all logic for running our hierarchical cross-validation scheme

[load_data.py](https://github.com/flavell-lab/SAAV_Decoding/blob/main/load_data.py): loads the relevant aligned data that is used in the modelling

[model.py](https://github.com/flavell-lab/SAAV_Decoding/blob/main/model.py): contains the Recurrent Neural Network (RNN) model description

[train.py](https://github.com/flavell-lab/SAAV_Decoding/blob/main/train.py): contains all training and evaluation logic

[util.py](https://github.com/flavell-lab/SAAV_Decoding/blob/main/util.py): contains any helper functions used for data structuring or augmentation
