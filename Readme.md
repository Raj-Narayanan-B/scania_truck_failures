
# Scania Truck Failures

This machine learning project addresses the prediction of Scania truck failures, specifically focusing on the Air Pressure System (APS). Utilizing a binary classification approach, the model distinguishes between failures caused by APS components and those originating from other factors. By leveraging the advantages of compressed air over hydraulic systems, the goal is to enhance predictive maintenance strategies, ensuring the reliability and longevity of heavy-duty vehicles.


## Project workflow and Execution

#### 1. Data Collection and Database Insertion:
* The Scania Truck Failure dataset comprises 60K records with 170 features, targeting the Air Pressure System (APS) failure prediction.
* Prepare the data for Cassandra database (Astra DB).
* Split data into batches columnwise due to free tier limitations. The free tire only allows 75 columns in each database.
* Upload data to the database with an added "ident_id" column. This is the Primary key. The databases should have a total of 75 columns with this primary key column included. This primary key is added to preserve the original order during data retrieval.

#### 2. Data Retrieval from Database:
* Three batches of training and testing data are retrieved.

#### 3. Initial Processing:
* Training data batches are merged into a single dataset based on the "ident_id" column.
* The target variable's name is corrected from "field_74_" to "class."

#### 4. Data Validation 1:
* Basic validation steps include:
    - mapping "neg" and "pos" to 0 and 1 respectively
    - replacing "na" with NaN
    - converting feature's datatypes to float.


#### 5. Data Validation 2:
* For training data, columns with:
    * more than 50% missing values
    * zero standard deviation 
    are dropped.
* Histogram features are saved for reference.



#### 6. Data Splitting:

* Validated training data is split into training (75%) and validation (25%) subsets.


#### 7. Final Processing:

* A pipeline is created for:
    * Imputation
    * Scaling
    Data imbalance handling using SMOTE-Tomek and dropping duplicate entries are addressed too.

#### 8. HP Tuning and Tracking:

* Hyperparameter tuning is performed on 12 machine learning models using Optuna.
    * The models used are:
        1. LogisticRegression
        2. SGDClassifier
        3. RandomForestClassifier
        4. AdaBoostClassifier
        5. GradientBoostingClassifier
        6. LGBMClassifier
        7. BaggingClassifier
        8. ExtraTreesClassifier
        9. HistGradientBoostingClassifier
        10. DecisionTreeClassifier
        11. XGBClassifier
        12. KNeighborsClassifier
* The Best_HP_Tuned_Model, StackingClassifier, and VotingClassifier are saved and versioned by Mlflow.

#### 9. Model Testing:

* The champion model is selected based on the highest accuracy among the challenger models.
* Predictions are made on the transformed test data, and the final test data prediction report is saved.

#### 10. Cloud Setup:

* The entire project is Dockerized, and workflows are created to push the image into AWS ECR.

#### 11. Data from User:

* Data is collected from trucks at a specified frequency.

#### 12. Fetch Champion Model:

* The champion model is fetched from MLflow and loaded into the system.

#### 13. Data Validation, Preprocessor Transformation, Prediction, Save Prediction in S3:

* A prediction pipeline is initiated to handle data from the Flask app, validating, transforming, predicting, and saving results in an S3 bucket.


## Installation

Install the requirements from the requirements.txt file

```bash
  pip install -r requirements.txt

```
    
## Running Tests

To run tests, run the following command

```bash
  pytest -v
```


## Authors

- [@Raj-Narayanan-B](https://github.com/Raj-Narayanan-B)

