# 引入必要的庫
import pandas as pd
from pycaret.classification import *

# 讀取Titanic資料集（假設你有一個CSV檔案）
df = pd.read_csv('train.csv')

# 初始化PyCaret環境，設置目標變量 'Survived'，並自動進行數據預處理
# 'target'是我們想預測的欄位，這裡是 'Survived'
clf = setup(data=df, target='Survived', session_id=123)

# 比較16個模型的表現
best_model = compare_models()

# 顯示最佳模型
print(best_model)

# 如果需要，可以進一步訓練並調整最佳模型
# 例如：對最佳模型進行超參數調整
tuned_model = tune_model(best_model)

# 最終評估並部署模型
evaluate_model(tuned_model)
