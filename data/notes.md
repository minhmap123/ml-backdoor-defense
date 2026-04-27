## Data Folder Structure
- `0_raw/`: Raw datasets. Contains subfolders for each dataset (IoTID20, CIC-IDS2017, CSE-CIC-IDS2018, CIC-IoT-2023) where raw files are placed.
- `1_processed/`: Preprocessed data. Contains subfolders for each dataset with processed files (e.g., scaled features, train/val/test splits). Output from preprocessing scripts.
- `artifacts/`: METADATA (e.g., fitted scalers, encoders, generated backdoor triggers).
- `notebooks/`: Jupyter notebooks for data exploration and preprocessing.

## General Preprocessing
- Data Cleaning
- normalization using the Quantile transformer
- scaling the data to the range of [-1, 1].

## Links
- [IoTID20](https://sites.google.com/view/iot-network-intrusion-dataset/home)
- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- [CIC-IoT-2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)