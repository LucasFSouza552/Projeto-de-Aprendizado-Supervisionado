# Relatório Final - Análise de Modelos de Classificação

#### 1. Introdução ao Problema:
Este projeto teve como objetivo principal a construção e avaliação de modelos de Machine Learning para a predição de diabetes. Priorizamos a análise do desempenho dos modelos, sua interpretabilidade e eficiência, além de documentar os desafios enfrentados ao longo do processo.

#### 2. Descrição do Dataset:
O dataset utilizado, carregado a partir de `data/diabetes.csv`, contém informações de saúde de pacientes, incluindo:
- **Pregnancies**: Número de gestações.
- **Glucose**: Nível de glicose.
- **BloodPressure**: Pressão arterial.
- **SkinThickness**: Espessura da prega cutânea tricipital.
- **Insulin**: Nível de insulina.
- **BMI**: Índice de Massa Corporal.
- **DiabetesPedigreeFunction**: Função de pedigree de diabetes.
- **Age**: Idade.
- **Outcome**: Variável alvo (0 = não diabético, 1 = diabético).

#### 3. EDA (Análise Exploratória de Dados) e Preparação dos Dados:
A análise exploratória e a preparação dos dados foram etapas cruciais para a qualidade do modelo. As principais etapas incluíram:

- **Verificação e Tratamento de Valores Inválidos:** Zeros foram substituídos por `NaN` em colunas como `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI`. Linhas com `NaN` foram removidas.
- **Análise de Distribuição e Outliers:** Histograma e boxplots foram utilizados para visualizar a distribuição das variáveis e detectar outliers. Outliers foram removidos usando o método IQR.
- **Análise de Correlações:** Uma matriz de correlação foi gerada, destacando `Glucose` e `Age` como as features mais correlacionadas com `Outcome`.
- **Normalização dos Dados:** `StandardScaler` foi aplicado às features numéricas.
- **Engenharia de Features:**
    - Criação de features de interação: `BMI_Age` e `Glucose_Insulin_Ratio`.
    - Categorização de variáveis: `BMI_Category`, `Age_Group` e `Glucose_Level`.
    - One-hot encoding para as variáveis categóricas.
- **Balanceamento da Variável Alvo:** O dataset foi balanceado (50% para cada classe) via undersampling para mitigar o desequilíbrio original (72.9% não diabético vs. 27.1% diabético).
- **Divisão Treino/Teste:** Os dados foram divididos em 80% para treino e 20% para teste.

#### 4. Descrição dos Modelos Implementados
Foram implementados e avaliados dois modelos de classificação:

- **Rede Neural Artificial (RNA)**:
    - **Arquitetura:** Sequencial, com camadas densas de 16, 8 e 1 neurônios, utilizando ativação ReLU nas ocultas e Sigmoid na saída.
    - **Otimizador:** Adam.
    - **Função de Perda:** Binary_crossentropy.
    - **Treinamento:** 50 épocas.

- **Random Forest Classifier**:
    - **Algoritmo:** Baseado em ensemble de árvores de decisão.
    - **Estimadores:** 100 estimadores.
    - **Balanceamento:** `class_weight='balanced'` utilizado para ajustar os pesos das classes.
    - **Reprodutibilidade:** `random_state=42`.

#### 5. Resultados e Comparação entre Modelos
As métricas de avaliação dos modelos foram as seguintes:

| Métrica       | Rede Neural Artificial (RNA) | Random Forest |
| :------------ | :--------------------------- | :------------ |
| **Acurácia** | 0.8000                       | **0.8286**|
| **Perda** | 0.4608                       | N/A           |
| **Precisão** | 0.8421                       | **1.0000**|
| **Recall** | **0.8000** | 0.7000|
| **F1-Score** | 0.8205                       | **0.8235**|
| **AUC-ROC** | **0.8833** | 0.8767|

### Análise Comparativa:

- **Melhor Desempenho**:
    - O **Random Forest** obteve maior **acurácia** (0.8286) e uma **precisão perfeita** (1.0000).
    - A **RNA** demonstrou um **recall superior** (0.8000) e um **AUC-ROC ligeiramente maior** (0.8833). Em problemas de saúde, o alto recall da RNA pode ser preferível para minimizar falsos negativos.

- **Interpretabilidade**:
    - O **Random Forest é mais interpretável**, permitindo a análise da importância das características.
    - A **RNA** é considerada uma "caixa preta", dificultando a compreensão direta do processo de predição.

- **Velocidade de Treinamento**:
    - O **Random Forest** foi **mais rápido de treinar**, não exigindo o processo iterativo de épocas da RNA.

- **Desafios Encontrados**:
    - **RNA:** Potencial para **overfitting** (acurácia de validação estagnou enquanto a de treino melhorava) e sua natureza de "caixa preta" para interpretabilidade.
    - **Random Forest:** Apesar do balanceamento de classes, o **recall menor** pode indicar dificuldade em identificar todos os casos positivos da classe minoritária.

#### 6. Conclusões Finais e Aprendizados do Grupo:
Este projeto reforçou a importância de cada etapa do pipeline de Machine Learning.

- **Random Forest**:
    - Apresentou uma performance robusta, especialmente em precisão e acurácia.
    - Oferece um bom equilíbrio entre desempenho e interpretabilidade.

- **Rede Neural (RNA)**:
    - Desempenho competitivo, com destaque para o recall e AUC-ROC.
    - Desafios relacionados ao overfitting e à complexidade de interpretação.

### Aprendizados do Grupo:
- A **preparação adequada dos dados** (tratamento de zeros/outliers, normalização) é fundamental para a qualidade do modelo.
- A **engenharia de características** pode enriquecer o dataset e impactar positivamente o desempenho.
- O **balanceamento de classes** é crucial em datasets desequilibrados para evitar viés do modelo.
- A **escolha do modelo** deve considerar não apenas a acurácia, mas também outras métricas (precisão, recall, F1-score, AUC-ROC) e a necessidade de interpretabilidade para o problema em questão.
- A **validação contínua** durante o treinamento (ex: `EarlyStopping` na RNA) é vital para garantir a generalização do modelo.