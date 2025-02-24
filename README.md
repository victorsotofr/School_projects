# School projects
**Some cool school projects during my year at École polytechnique**

---

## recommender_system  
**GOAL**: build a simple recommender system using the [MovieLens100k dataset](https://grouplens.org/datasets/movielens/100k/).
  * **Data**: [MovieLens100k dataset](https://grouplens.org/datasets/movielens/100k/)
  * **Libraries**: pandas numpy scipy surprise
  * **Cool methods**: SVD, weighted SVD

---

## reinforcement_learning  
**GOAL**: show how to optimize an agent using reinforcement learning.
!! Uses the *RL.py* file !!
   * **Environment**: Taxi-v3 environment.
   * **Libraries**: numpy gymnasium
   * **Cool methods**: WIP

---

## pytorch
**GOAL**
   * **Data**: [PyTorch MNIST Dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
   * **Libraries**: numpy pandas matplotlib torch
   * **Cool methods**: NN construction

---

## german_credit
**GOAL**: build the best possible classifier for a default risk, with a non classical metric
   * **Credits**: Erwan Le Pennec, Professor @École polytechnique
   * **Data**: [Kaggle Challenge](https://www.kaggle.com/competitions/dsb-24-german-credit/overview)
   * **Libraries**: numpy pandas matplotlib seaborn sklearn
   * **Cool methods**: optuna hyperparameters optimization using StratifiedKFold (for unbalanced classes). Optimization was done on GoogleColab.

Here is the custom business-oriented metric:
```
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
```

---

## WIP

---

## How to Use  
1. Clone the repository:  
   ```bash
   git clone https://github.com/victorsotofr/School_projects.git
   cd School_projects
