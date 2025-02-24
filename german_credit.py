# ******************************************************************************
# KAGGLE CHALLENGE - GERMAN CREDIT - DSB 2025
# Professor: Erwan Le Pennec
# Goal: In this competition, you have to build the best possible classifier for a default risk… with a non classical metric!
# Challenge available [here](https://www.kaggle.com/competitions/dsb-24-german-credit/overview)
# ******************************************************************************

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ******************************************************************************
# TAILOR-MADE EVALUATION FUNCTION
# ******************************************************************************
# The following evaluation function is provided in the Kaggle challenge:
def compute_costs(LoanAmount):
     return({'Risk_No Risk': 5.0 + .6 * LoanAmount, 'No Risk_No Risk': 1.0 - .05 * LoanAmount,
         'Risk_Risk': 1.0, 'No Risk_Risk': 1.0})

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
   '''
   A custom metric for the German credit dataset
   '''
   real_prop = {'Risk': .02, 'No Risk': .98}
   train_prop = {'Risk': 1/3, 'No Risk': 2/3}
   custom_weight = {'Risk': real_prop['Risk']/train_prop['Risk'], 'No Risk': real_prop['No Risk']/train_prop['No Risk']}
   costs = compute_costs(solution['LoanAmount'])
   y_true = solution['Risk']
   y_pred = submission['Risk']
   loss = (y_true=='Risk') * custom_weight['Risk'] *\
               ((y_pred=='Risk') * costs['Risk_Risk'] + (y_pred=='No Risk') * costs['Risk_No Risk']) +\
            (y_true=='No Risk') * custom_weight['No Risk'] *\
               ((y_pred=='Risk') * costs['No Risk_Risk'] + (y_pred=='No Risk') * costs['No Risk_No Risk'])
   return loss.mean()

# ******************************************************************************
# MY WORK
# ******************************************************************************

train = pd.read_csv("./data/dsb-24-german-credit/german_credit_train.csv")
test = pd.read_csv("./data/dsb-24-german-credit/german_credit_test.csv")

numerical_features = train.select_dtypes(include='number').columns

# we extract the categorical columns' names from the test set to avoid having the 'Risk' (target) column included
categorical_features = test.select_dtypes(exclude='number').columns

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier( # We fine-tuned the Hyperparameters using Optuna and StratifiedKFold on Google Colab
        n_estimators = 200, 
        max_depth = 11, 
        min_samples_split = 6, 
        min_samples_leaf = 4,
        random_state =42
        )
)

X = train.drop(columns=['Risk'])
y = train['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

solution = pd.DataFrame({
        'Id': X_test.index, 
        'Risk': y_test,
        'LoanAmount': X_test['LoanAmount'].values
    })

submission = pd.DataFrame({
    'Id': X_test.index, 
    'Risk': y_pred,
    'LoanAmount': X_test['LoanAmount'].values
    })

loss = score(solution, submission, row_id_column_name='Id')
print(f'Custom loss is {loss}')

pipeline.fit(X, y)
y_final = pipeline.predict(test)

final_submission = pd.DataFrame({
    'Id': test.index,
    'Risk': y_final,
    'LoanAmount': test['LoanAmount'].values
})

final_submission.to_csv('submission.csv', index=False)