import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder

# Instanciar o codificador
encoder = LabelEncoder()

# Carregar os datasets
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

# Inicializar o conjunto de melhores features e a melhor F1-score
best_features = []
best_f1_score = 0

# Lista de todas as características
all_features = list(X_train.columns)

for i in range(len(all_features)):
    print("Rodada ", i + 1)
    # Selecionar a próxima feature a ser adicionada
    next_feature = all_features[i]

    # Criar um novo conjunto com as melhores features + a próxima feature
    current_features = best_features + [next_feature]
    print("Features atuais: ", current_features)

    # Criar novos dataframes apenas com as features atuais
    X_train_selected = X_train[current_features]
    X_test_selected = X_test[current_features]

    before = time.time()

    # Treinar um novo modelo com as features selecionadas
    model_selected = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_selected.fit(X_train_selected, y_train)

    # Fazer previsões com o modelo ajustado
    y_pred_selected = model_selected.predict(X_test_selected)

    after = time.time()

    # Avaliar o modelo ajustado
    f1 = f1_score(y_test, y_pred_selected)
    recall = recall_score(y_test, y_pred_selected)
    precision = precision_score(y_test, y_pred_selected)


    print({f1}, ",", {recall}, ",", {precision}, ",", {after - before})
    print(current_features)

    # Se o F1 Score melhorar, atualizar o melhor conjunto de features
    if f1 > best_f1_score or f1 == 0.0:
        best_f1_score = f1
        best_features = current_features
