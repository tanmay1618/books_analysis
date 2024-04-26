**Setup:**

1. Run command "start-master.sh"
2. Go to http://localhost:8080 and copy the url
3. Run command "start-worker.sh <url>"
4. Run command "virtualenv -p python3 venv"
5. Run "source venv/bin/activate"
6. Run "pip install -r requirements.txt"


**Process Followed:**
1. Preliminary EDA was done on the notebooks. The notebooks contains ipynb files for the analysis.
2. The EDA analysis is present in the two files:
    2.1 EDA1.pdf
    2.2 EDA2.pdf
3. The model building is divided into two parts:
    3.1 Feature Engineering: All the features are created in the feature_engineering.py file. The final file features.csv is produced which can be used in the model building.
       For feature engineering,
       <li> Title - Tfidf and sentence embedding is done </li>
       <li> Description - Tfidf and sentence embedding is done </li>
       <li> cateories - One Hot vector is done </li>
       <li> publishedDate - publishedMonth and publishedYear is calculated </li>
       <li> authors - tfidf is done. (TBD: Label Encoding can also be done here) </li>
       <li> publisher - tfidf is done (TBD: Label Encoding and one-hot vector can be done here) </li>
    3.2 Model Training: The features.csv is loaded in experiements.py and split into training and test. Each model is trained and measured on mse/mape on the test set.
5. Experiments: Experiment management is done in through python code here. Models are saved as pickle files.(TBD: Replace experiment management with a better library). Also, feature selection can be done here for different models.
   4.1 Experiment 1: Linear Regression
   4.2 Experiment 2: Random Forest Regresor
   4.3 Experiment 3: Neural Network Model
