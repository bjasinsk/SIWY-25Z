#!/bin/bash

set -e  # Exit on error

DATA_DIR=${PWD}/data
RAW_DATA_DIR=${DATA_DIR}/raw

# Create directories if they don't exist
mkdir -p ${RAW_DATA_DIR}

echo "Downloading datasets to ${RAW_DATA_DIR}"

# Download commercial aircraft dataset
echo "Downloading commercial-aircraft-dataset..."
curl -L -o ${RAW_DATA_DIR}/commercial-aircraft-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/nelyg8002000/commercial-aircraft-dataset

# Extract commercial aircraft dataset
echo "Extracting commercial-aircraft-dataset..."
unzip -q -o ${RAW_DATA_DIR}/commercial-aircraft-dataset.zip -d ${RAW_DATA_DIR} || {
    echo "Warning: Extraction failed or already extracted"
}

# Download dog-and-cat dataset
echo "Downloading dog-and-cat-classification-dataset..."
curl -L -o ${RAW_DATA_DIR}/dog-and-cat-classification-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset

# Extract dog-and-cat dataset
echo "Extracting dog-and-cat-classification-dataset..."
unzip -q -o ${RAW_DATA_DIR}/dog-and-cat-classification-dataset.zip -d ${RAW_DATA_DIR} || {
    echo "Warning: Extraction failed or already extracted"
}

echo "Done!"