# HousePrices

## Kaggle-ის კონკურსის მოკლე მიმოხილვა  
ამ კონკურსის მიზანი იყო საცხოვრებელი სახლების ფასების პროგნოზირება. მონაცემთა ანალიზისა და მოდელების გამოყენებით, უნდა შეგვედგინა მაქსიმალურად ზუსტი პროგნოზი `SalePrice` ცვლადისთვის.

## თქვენი მიდგომა პრობლემის გადასაჭრელად  
პირველი ეტაპზე გავწმინდე მონაცემები, მოვახდინე Null მნიშვნელობების დამუშავება, კატეგორიული ცვლადების გარდაქმნა, მაღალკორელირებული სვეტების მოცილება და დაბალი ვარიაციის სვეტების დაგდება. გავატარე ექსპერიმენტები სხვადასხვა ალგორითმზე და საბოლოოდ DecisionTreeRegressor გამოვიყენე, k-fold validation-ით და MLflow-ით ვთვალე შედეგები.

---

## რეპოზიტორიის სტრუქტურა

```
HousePrices/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── notebooks/
│   ├── model_experiment.ipynb
│   ├── model_inference.ipynb
│
├── pipeline/
│   ├── preprocessors.py
│   ├── transformers.py
│
├── mlruns/  ← MLflow experiments
│
├── README.md
└── requirements.txt
```

---

## ყველა ფაილის განმარტება

- `model_experiment.ipynb`: მთავარი ექსპერიმენტების სამუშაო რვეული
- `model_inference.ipynb`: საბოლოო მოდელით პროგნოზის გაკეთება
- `preprocessors.py`: მონაცემთა გაწმენდის და feature engineering კლასები
- `transformers.py`: Custom transformer-ები Pipeline-ში გამოსაყენებლად
- `requirements.txt`: ყველა საჭირო პაკეტის ჩამონათვალი
- `mlruns/`: MLflow-ის მიერ ჩაწერილი ექსპერიმენტების ფოლდერი

---


## Cleaning

პირველ რიგში, დავიწყე preprocessing ნაწილი და გადავარჩიე `cat_columns` და `num_columns`. თითოეულისათვის დავწერე ცალკე pipeline კლასი (`cat_transformer_pipeline` და `num_transformer_pipeline`), რომელიც მარტივად ცვლილებადია.

### Nan მნიშვნელობების დამუშავება

პირველ ცდისთვის:
- კატეგორიული სვეტები → ჩავანაცვლე mode-ით
- რიცხვითი სვეტები → ასევე ჩავანაცვლე mode-ით

დავდროფე სვეტები, რომლებიც შეიცავდა > 80% NA-ს:
```
Dropped columns with >80.0% missing values:
 ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
```

დავდროფე სვეტები, რომლებიც > 95% ერთსა და იმავე მნიშვნელობას შეიცავდნენ:
```
Dropped low-variance columns with >95.0% same value:
 ['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'LowQualFinSF', 'KitchenAbvGr', 'GarageQual', 'GarageCond', '3SsnPorch', 'PoolArea', 'MiscVal']
```

---

## Feature Engineering

მონაცემთა გაწმენდის შემდეგ, დარჩა 64 ცვლადი. შემდეგი ნაბიჯები:

### კატეგორიული ცვლადების რიცხვითში გადაყვანა  
ONE-HOT ENCODING გავუკეთე იმ ცვლადებს, რომლებშიც ≤ 3 უნიკალური მნიშვნელობა იყო. დანარჩენზე გამოვიყენე WOEEncoder.

### Feature Selection  
- გავაკეთე კორელაციის ფილტრი (threshold > 0.8)
- დავიტოვე სვეტი, რომელსაც მაღალი კორელაცია ჰქონდა target-თან  
📌 იხილეთ შედეგი MLflow-ზე:  
https://dagshub.com/ekvirika/HousePrices.mlflow/#/experiments/1/runs/d3f840dc06ff42599ec77adcc8f10260

---

## Training

### ტესტირებული მოდელები  
- Linear Regression (baseline)
- Decision Tree Regressor (საბოლოო შერჩეული მოდელი)

### Hyperparameter ოპტიმიზაციის მიდგომა  
ამ ეტაპზე არ გავუკეთე ჰიპერპარამეტრების ოპტიმიზაცია — DecisionTree მოდელი გაშვებულია default პარამეტრებით. მომდევნო ნაბიჯებში იგეგმება GridSearchCV ან Optuna ინტეგრაცია.

---

## MLflow Tracking

მოდელის ყველა ექსპერიმენტი ჩავწერე MLflow-ში.

📌 [MLflow ექსპერიმენტების ბმული](https://dagshub.com/ekvirika/HousePrices.mlflow)

### ჩაწერილი მეტრიკები:
- MAE
- MSE
- RMSE
- R²
- RMSE (log-scale, როგორც კონკურსის შეფასება მოითხოვდა)

### საუკეთესო მოდელის შედეგები (K-Fold Cross Validation-ის საშუალებით):
- mean_rmse_log: `0.1655`
- Kaggle public score: `0.17430`

---
