## Environment configuration

```
python3.8
pytorch-transformers==1.0.0
torch==1.3.0a0+24ae9b5
```

## Dataset

The original data set downloaded is placed in the `$DOWNLOAD_PATH` folder, and the processed data set is placed in the `$TARGET_PATH` folder.

## Download and Preprocessing data

To download the MultiWOZ dataset and preprocess it, please run this script first.
You can choose the version of the dataset. ('2.1', '2.0')
The downloaded original dataset will be located in `$DOWNLOAD_PATH`, and after preprocessing, it will be located in `$TARGET_PATH`.

```
python3 create_data.py --main_dir $DOWNLOAD_PATH --target_path $TARGET_PATH --mwz_ver '2.1' # or '2.0'
```

## Train

Execute the following command to start training:


```
python3 train.py --data_root $DATASET_DIR --save_dir $SAVE_DIR --bert_ckpt_path `bert-base-uncased-pytorch_model.bin --op_code '4'`
```

## Evaluate

Execute the following command to evaluate the model:

```
python3 evaluation.py --model_ckpt_path $MODEL_PATH --data_root $DATASET_DIR
```

