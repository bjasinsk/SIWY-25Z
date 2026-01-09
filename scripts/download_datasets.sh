#!/bin/bash

set -e  # Exit on error

DATA_DIR=${PWD}/data
RAW_DATA_DIR=${DATA_DIR}/raw

# Create directories if they don't exist
mkdir -p ${RAW_DATA_DIR}

echo "Downloading datasets to ${RAW_DATA_DIR}"

# Function to generate Google Drive download link
get_gdrive_link() {
    local file_id=$1
    echo "https://drive.usercontent.google.com/download?id=${file_id}&confirm=xxx"
}

# Function to download and unzip a file
download_and_unzip() {
    local name=$1
    local path=$2
    local link=$3
    local zip_file="${path}/${name}.zip"

    if [ -f "${zip_file}" ]; then
        echo "File ${zip_file} already exists. Skipping download."
    else
        echo "Downloading ${name}..."
        curl -L -o "${zip_file}" "${link}"
    fi

    echo "Extracting ${name}..."
    unzip -q -o "${zip_file}" -d "${path}" || {
        echo "Warning: Extraction failed or already extracted"
    }
}

# Download commercial aircraft dataset
download_and_unzip "commercial-aircraft-dataset" "${RAW_DATA_DIR}" \
    "https://www.kaggle.com/api/v1/datasets/download/nelyg8002000/commercial-aircraft-dataset"

# Download dog-and-cat dataset
download_and_unzip "dog-and-cat-classification-dataset" "${RAW_DATA_DIR}" \
    "https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset"

# Download bus-and-truck (task2)dataset
download_and_unzip "bus-and-track" "${RAW_DATA_DIR}" \
    "$(get_gdrive_link '1B01c3KJZMHJQZ-QkysonH-DUIdMS5niZ')"

# Download horse-and-elephant (task3)dataset
download_and_unzip "horse-and-elephant" "${RAW_DATA_DIR}" \
    "$(get_gdrive_link '1eh1soLM9TroU5gx4KPEeNQ9CZMn3ihVV')"

echo "Done!"
