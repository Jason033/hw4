from pycaret.classification import *
import pandas as pd

# 加載資料集
data = pd.read_csv('train.csv')

# 設置pycaret環境
clf = setup(data, target='Survived', session_id=123)
# 在 setup 函數中，PyCaret會自動處理基本的特徵工程
# 如數值和類別型特徵的處理、缺失值處理等
clf = setup(data, target='Survived', session_id=123, 
            numeric_features=['Age', 'Fare'],  # 可以指定數值型特徵
            categorical_features=['Sex', 'Embarked', 'Pclass'])  # 類別特徵
# 比較不同的模型，並選擇集成方法
best_model = compare_models()
# 使用隨機森林進行模型訓練
rf_model = create_model('rf')

# 使用XGBoost進行模型訓練
xgb_model = create_model('xgboost')

# 使用LightGBM進行模型訓練
lgbm_model = create_model('lightgbm')
# 使用Optuna進行超參數優化
tuned_rf_model = tune_model(rf_model)  # 對隨機森林進行優化
tuned_xgb_model = tune_model(xgb_model)  # 對XGBoost進行優化
tuned_lgbm_model = tune_model(lgbm_model)  # 對LightGBM進行優化
# 堆疊多個經過調參的模型
stacked_model = stack_models([tuned_rf_model, tuned_xgb_model, tuned_lgbm_model])

# 評估堆疊模型的表現
evaluate_model(stacked_model)
# 評估經過優化的模型
evaluate_model(tuned_rf_model)
evaluate_model(tuned_xgb_model)
evaluate_model(tuned_lgbm_model)

# 評估堆疊模型的表現
evaluate_model(stacked_model)
# 將最優模型保存
save_model(stacked_model, 'best_titanic_model')
