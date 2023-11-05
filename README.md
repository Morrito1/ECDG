## 环境配置

```
python3.8
pytorch-transformers==1.0.0
torch==1.3.0a0+24ae9b5
```

## 数据集

下载的原始数据集放置在`$DOWNLOAD_PATH`文件夹下 ，处理完的数据集放在文件夹 `$TARGET_PATH`下.

## 训练

执行如下命令开始训练：


```
python3 train.py --data_root $DATASET_DIR --save_dir $SAVE_DIR --bert_ckpt_path `bert-base-uncased-pytorch_model.bin --op_code '4'`
```

## 评估

执行如下命令对模型进行评估：

```
python3 evaluation.py --model_ckpt_path $MODEL_PATH --data_root $DATASET_DIR
```

