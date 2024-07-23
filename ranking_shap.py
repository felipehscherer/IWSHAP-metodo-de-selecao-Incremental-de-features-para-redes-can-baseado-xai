import time

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder

# Instanciar o codificador
encoder = LabelEncoder()

# Carregar os datasets
# df1 é o dataset sem ataque e o df2 com ataque
df_safe = pd.read_parquet('repository/dump6.parquet')
df_attack = pd.read_parquet('repository/dump6-susp-2B0h.parquet')

# Combinar os datasets
df = pd.concat([df_safe, df_attack], ignore_index=True)

# Codificar colunas do tipo object
df['112_CUR_GR'] = encoder.fit_transform(df['112_CUR_GR'])

# Dividir os dados novamente após a codificação
X = df.drop('label', axis=1)
y = df['label']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar e treinar o modelo XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Treinar o modelo
model.fit(X_train, y_train)

# Calcular os valores SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
"""
# Plotar os valores SHAP para a importância das features
shap.summary_plot(shap_values, X_train)
"""


# Obter a importância das features
feature_importance = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

for i in range(1, 688):
    print("rodada ", i)
    # Selecionar as features mais importantes (por exemplo, as top 10 mais importantes)
    top_features = feature_importance_df['feature'].head(i).tolist()

    print(top_features)

    # Criar um novo dataframe apenas com as features selecionadas
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    before = time.time()

    # Treinar um novo modelo com as features selecionadas
    model_selected = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_selected.fit(X_train_selected, y_train)

    # Fazer previsões com o modelo ajustado
    y_pred_selected = model_selected.predict(X_test_selected)

    after = time.time()

    # Avaliar o modelo ajustado
    acuracia = f1_score(y_test, y_pred_selected)
    recall = recall_score(y_test, y_pred_selected)
    precision = precision_score(y_test, y_pred_selected)

    print(acuracia, ",", recall, ",", precision, ",", (after - before))

"""
['2B0_MsgCount', '2B0_SAS_Angle', '2B0_CheckSum', '5B0_CF_Clu_Odometer', '260_AliveCounter', '220_ESP12_Checksum', '080_CF_Ems_Alive', '164_CF_Esc_AliveCnt', '111_CF_Tcu_Alive1', '50C_CF_Clu_DTE']
F1 Score: 0.8861093519104919
Recall: 0.8744791666666667
Precision: 0.8980530594779632
"""
