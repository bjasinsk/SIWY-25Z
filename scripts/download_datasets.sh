#!/bin/bash

DATA_DIR=${PWD}/data
RAW_DATA_DIR=${DATA_DIR}/raw

echo "Downloading datasets to ${RAW_DATA_DIR}"

curl -L -o ${RAW_DATA_DIR}/commercial-aircraft-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/nelyg8002000/commercial-aircraft-dataset

curl -L -o ${RAW_DATA_DIR}/dog-and-cat-classification-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset
