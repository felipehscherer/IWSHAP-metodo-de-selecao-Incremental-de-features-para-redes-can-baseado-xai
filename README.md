# README

## Introdução

O **IWSHAP** (Iterative Wrapper Subset Selection with SHAP) é um algoritmo desenvolvido para a seleção de características em redes CAN (Controller Area Network), combinando os valores SHAP (SHapley Additive exPlanations) com o algoritmo **Iterative Wrapper Subset Selection (IWSS)**. Este método visa otimizar a qualidade da seleção de características ao mesmo tempo em que minimiza o consumo de recursos computacionais, um aspecto crucial para redes CAN que operam com processamento de mensagens em tempo real.

## Descrição do Algoritmo

O algoritmo IWSHAP é detalhado no pseudo-código apresentado abaixo. Ele segue uma abordagem iterativa para selecionar as características mais relevantes com base nos valores SHAP, que quantificam a importância de cada característica no modelo de classificação.

### Pseudo-código

```pseudo
1. (X_train, X_test, y_train, y_test) <- train_test_split(X, y, 0.2, 42)
2. model <- XGBoost(use_label_encoder=0, eval_metric='logloss')
3. model.fit(X_train, y_train)
4. shap_values <- shap.TreeExplainer(model).shap_values(X_train)
5. importancia_df <- np.abs(shap_values).mean(0)
6. importancia_df <- pd.DataFrame({'feature': X_train.columns, 'importance': importancia})
7. importancia_df <- importancia_df.sort_values('importance', ascending=0)
8. best_features <- {}, best_f1_score <- 0
9. for i <- 1 to len(importancia_df):
10.     current_features <- best_features ∪ {importancia_df['feature'][i]}
11.     X_train_selected, X_test_selected <- X_train[current_features], X_test[current_features]
12.     model_selected <- XGBoost(use_label_encoder=0, eval_metric='logloss')
13.     model_selected.fit(X_train_selected, y_train)
14.     y_pred_selected <- model_selected.predict(X_test_selected)
15.     f1 <- f1_score(y_test, y_pred_selected)
16.     if f1 > best_f1_score or f1 == 0:
17.         (best_f1_score, best_features) <- (f1, current_features)
18. (best_features_final, best_f1_score_final) <- (best_features, best_f1_score)
```

### Etapas do Algoritmo
Divisão dos Dados:

* Os dados são divididos em conjuntos de treinamento (80%) e teste (20%), com o random_state definido como 42 para garantir a reprodutibilidade dos resultados.
Treinamento Inicial do Modelo:

* Um modelo XGBoost é instanciado e treinado com o conjunto de dados de treinamento.
Cálculo dos Valores SHAP:

* Utiliza-se o shap.TreeExplainer para calcular os valores SHAP, que quantificam a importância das características no modelo.
Criação do DataFrame de Importância:

* Cria-se um DataFrame com as características e suas respectivas importâncias, ordenado de forma decrescente.
Iteração e Seleção de Características:

* Inicia-se com um conjunto vazio de características e a melhor pontuação de F1-Score definida como zero.
* Itera-se sobre as características ordenadas, adicionando cada uma ao conjunto atual e avaliando a performance do modelo com este novo conjunto.
* O conjunto de características é atualizado apenas se a nova pontuação de F1-Score for superior à anterior, ou se for a primeira iteração (considerada como base).
Comparação e Atualização:

* Se a pontuação F1-Score do conjunto atual for superior à anterior, ou se for a primeira iteração, o conjunto de características e a pontuação são atualizados.
Resultado Final:

* Após iterar sobre todas as características, obtém-se o conjunto final das melhores características e a melhor pontuação de F1-Score, proporcionando uma seleção eficiente e balanceada.

### Configuração de ambiente

Versão do python utilizada: 3.12
```
pip install numpy
pip install pandas
pip install shap
pip install xgboost
pip install scikit-learn
```
Para manipular o arquivo .parquet sugiro: 
```
pip install fastparquet
```

### Aplicações
O IWSHAP é particularmente útil para a detecção de intrusões em redes CAN, onde a eficiência computacional e a precisão são críticas. A combinação dos valores SHAP com o IWSS permite uma seleção de características que melhora a performance do modelo de classificação enquanto reduz o consumo de recursos.

### Conclusão
O IWSHAP é uma poderosa ferramenta para a seleção de características, especialmente em ambientes de tempo real como redes CAN. Seu uso de técnicas avançadas de XAI (Explainable AI) e métodos iterativos de seleção garante uma solução robusta e eficiente para problemas complexos de classificação.
