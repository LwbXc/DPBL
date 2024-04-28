# Uberl: Denoised Universal User Behavior Representation Learning

Our Python version is 3.5.2 and torch version is 1.2.0.

## Introduction
This repository is the reporisity of **Uberl: Denoised Universal User Behavior Representation Learning** (currently under submission). Uberl is an unsupervised representation learning model to learning informative and high-quality representations for noisy user behavior sequences.

## Datasets
+ By default, our dataloader will load the data from CSV files as the training dataset or test dataset, and the format of each line in these files is as follows:

  ~~~
  user_id, date, timestamp1, action1, timestamp2, action2, timestamp3, action3, ...
  ~~~
  where each action is represented as an integer.

+ If you want to load your data in other formats, you need to preprocess the data into the format we specify, or implement your own dataloader by editing `./uberl/dataset/dataset_test.py` and `./uberl/dataset/dataset_train.py`.

## Quick Demos
1. Put your dataset under `./data`
2. We provide following script to train the model:
    ```bash
    bash ./scripts/train.sh
    ``` 
3. For inference, we provide following script to generate representations:
    ```bash
    bash ./scripts/inference.sh
    ```
4. After pre-training, you can run following script to train classifier for downstream tasks:
   ```bash
   bash ./scripts/downstream_tasks.sh
   ```

## Detailed Usage

### Train the model
+ For example, if we want to train Uberl on /data/dataset/train.csv, and save the model per epoch as /data/save/uberl_ep0, /data/save/uberl_ep1, ...
~~~bash
python main.py --train_dataset /data/dataset/train.csv --model_save /data/save --model_name uberl --preference_num 16 --epochs 100 --batch_size 64
~~~

### Inference for the model
+ For example, if we want to load /data/save/uberl_ep0 and generate embeddings for /data/dataset/test.csv:
~~~
python main.py --train_mode 2 --test_dataset /data/dataset/test.csv --model_save /data/save --load_file uberl_ep0 --preference_num 16 --epochs 100 --batch_size 64
~~~

### Train for downstream tasks

+ Take anomaly detection as an example, we just train a classifier to predict the label. For example, if the embeddings of training dataset and test dataset are /data/dataset/train_embeddings.json and /data/dataset/test_embeddings.json, and labels of training dataset and test dataset are /data/dataset/train_labels.json and /data/dataset/test_labels.json:
~~~
python classifier/classifier.py --train_embeddings /data/dataset/train_embeddings.json --test_embeddings /data/dataset/test_embeddings.json --train_labels /data/dataset/train_labels.json --test_labels /data/dataset/test_labels.json --epochs 100 --result_save /data/logs/anomaly_detection.txt
~~~
