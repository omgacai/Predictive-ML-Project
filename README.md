# aiip5-Yong-Sook-Mun-124H
### Full name: Yong Sook Mun
### Email address: yongsm1223@gmail.com


# Folder Structure 
```plaintext
├── data/                             # Not pushed to git
│   ├── agri.db                       # The raw dataset used for analysis (Not pushed to git)
├── eda.ipynb                         # Exploratory Data Analysis (EDA) Jupyter Notebook
├── src/
│   ├── data_loader.py                # Load data
│   ├── data_cleaning.py              # Clean data
│   ├── type_stage_merger.py          # Script to form the new column plant_type_stage
│   ├── preprocess.py                 # Preprocessing data 
│   ├── model.py                      # Loading models 
│   ├── main.py                       # Main script to train and test the model
│   ├── config.py                     # Load config.yaml 
├── config.yaml                       # Configuration file with hyperparameters and model parameters
├── results/
│   ├── model_performance.csv         # File containing model evaluation results
├── README.md                        
├── run.sh                      
└── requirements.txt
```
# Instructions for executing pipeline
## Prerequisites
Before running the pipeline, ensure that the required libraries are installed. You can install the dependencies using `pip`:
```
pip install -r requirements.txt
```

## Running the pipeline
1. **Data Preperation:** create a folder 'data' and place the raw dataset, agri.db inside
2. **Preprocessing and Feature Engineering:** Default parameters are set but you can modify them in the config.yaml file
3. Executing model: head into the main folder directory and run this in your terminal
```
run.sh
```

