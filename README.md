**Setup:**

1. Run command "start-master.sh"
2. Go to http://localhost:8080 and copy the url
3. Run command "start-worker.sh <url>"
4. Run command "virtualenv -p python3 venv"
5. Run "source venv/bin/activate"
6. Run "pip install -r requirements.txt"


**Process Followed:**
1. Preliminary EDA was done on the notebooks. The notebooks contain `.ipynb` files for the analysis.
2. The EDA analysis is present in the two files:
    - EDA1.pdf
    - EDA2.pdf
3. The model building is divided into two parts:
    3.1 Feature Engineering: All the features are created in the `feature_engineering.py` file. The final file `features.csv` is produced which can be used in the model building.
        - For feature engineering:
            - Title: TF-IDF and sentence embedding are done.
            - Description: TF-IDF and sentence embedding are done.
            - Categories: One-hot vector is done.
            - Published Date: Published month and published year are calculated.
            - Authors: TF-IDF is done. (To be determined: Label Encoding can also be done here)
            - Publisher: TF-IDF is done. (To be determined: Label Encoding and one-hot vector can be done here)
    3.2 Model Training: The `features.csv` is loaded in `experiments.py` and split into training and test sets. Each model is trained and measured on MSE/MAPE on the test set.
4. Experiments: Experiment management is done through Python code here. Models are saved as pickle files. (To be determined: Replace experiment management with a better library). Also, feature selection can be done here for different models.
    4.1 Experiment 1: Linear Regression
    4.2 Experiment 2: Random Forest Regressor
    4.3 Experiment 3: Neural Network Model



