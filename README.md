
# About Project

This project was developed to identify potential churn customers at an early stage.  
By analyzing customer behavior, the model predicts whether a customer is likely to churn, helping businesses take actions.

# Installation

First clone the repository:

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

# Usage

Run the data processing script:

```bash
python data_processing.py
```
Then, run the model training script:

```bash
python Xgb-train.py
```
Finally, run the evaulation script:

```bash
python Xgb-evaluate.py
```

# How It Works
The data processing script processes raw data and applies all necessary data transformation steps.  
After execution, the processed datasets are saved in the `Datasets/` directory as:

- `train.csv`  
- `test.csv`  
- `validation.csv` 

Once the processing script is executed, all preprocessing steps are completed, and the datasets are ready for model training.

After data processing, the model training script is used to train the XGBoost model.  
Once the training is complete, the trained model files are saved in the `Model/` directory as:

- `xgb_model.pkl`  

After training the XGBoost model, the evaluation script is used to assess the model's performance.  
This script calculates key evaluation metrics and displays the results, including:

- **Accuracy**
- **F1 Score**
- **Recall**
- **Precision**
- **Confusion Matrix**

By running the evaluation script, you can analyze how well the model performs on the test dataset.

If you have a custom dataset for testing the model, you can run the Custom_predictions.py script.

If you want to create a Docker container, you can find the Dockerfile within the project files.

```bash
docker build -t churn-evaluate .
```

Run container

```bash
docker run --rm churn-evaluate
```

# Project Structure
```
Case-Study-Teknasyon/
│── Datasets/                # Dataset files
│   ├── churn_data.csv       # Raw dataset
│   ├── test.csv             # Processed test dataset
│   ├── train.csv            # Processed train dataset
│   ├── validation.csv       # Processed validation dataset
│
│── Model/                   # Trained model files
│   ├── xgb_model.pkl        # Saved XGBoost model

│── Utils/                   
│
│── data_processing.py       # Data processing script
│── functions.py             # Utility functions for data processing
│── Xgb-evaluate.py          # Model evaluation script
│── Xgb-train.py             # Model training script
│── Custom_predictions.py    # Model evaluation script for custom dataset 
│── Dockerfile               # for Docker
│── README.md                # Project documentation
│── requirements.txt         # Python dependencies list
```