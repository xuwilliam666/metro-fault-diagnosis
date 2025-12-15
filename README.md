# Metro Fault Diagnosis
This repository contains code and data used for constructing 
the framework for the metro fault diagnosis model.

## Project Structure
```text
├── data
│   ├── processed
│   │   └── bogie
│   │       ├── dataset1_test.csv
│   │       ├── dataset1_train.csv
│   │       └── dataset1_val.csv
│   └── raw
│       └── bogie
│           ├── dataset1.csv
│           └── dataset2.csv
├── notebooks
├── requirements.txt
├── scripts                        # data processing scripts
│   └── split_bogie_dataset1.py
└── src
    ├── __init__.py
    ├── data
    │   └── bogie_dataset.py
    └── models
        └── lstm_fcn.py
```
