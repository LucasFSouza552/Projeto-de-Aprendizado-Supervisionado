# Relatório Final - Análise de Modelos de Classificação

#### 1. Introdução ao Problema:
Este trabalho teve como objetivo aplicar e comparar diferentes algoritmos de classificação em um conjunto de dados real, visando a classificação correta das classes alvo. Tendo como prioridade analisar o desempenho dos modelos, sua interpretabilidade e eficiência. Entender os desafios enfrentados durante todo o processo.

#### 2. Descrição do Dataset:
O dataset utilizado é Pima Indians Diabetes e ele possui as seguintes características: 
- Aproximadamente 768 linhas
- 8 variáveis preditoras
- Variável alvo: Outcome (0 = não diabético, 1 = diabético)
- Atributos: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction e Age.

#### 3. EDA (Análise Exploratória de Dados) e Preparação dos Dados: 
Ao fazer a análise exploratória dos dados aplicamos as seguintes etapas, em busca da validação, tratamento e normatização dos dados:

- Verificação de valores ausentes e substituição de zeros por NaN
- Análise de distribuição das variáveis com histogramas e KDE
- Análise de correlações entre atributos usando matriz de correlação
- Detecção e tratamento de outliers usando método IQR
- Normalização dos dados usando StandardScaler
- Engenharia de features:
  - Criação de features de interação (BMI_Age, Glucose_Insulin_Ratio)
  - Categorização de variáveis (BMI_Category, Age_Group, Glucose_Level)
  - One-hot encoding para variáveis categóricas
- Balanceamento da variável alvo quando necessário
- Divisão do conjunto em treino e teste (80/20)

#### 4. Descrição dos Modelos Implementados
Foram implementados e avaliados dois modelos principais:

- **Random Forest**: 
  - Conjunto de árvores de decisão treinados de forma aleatória
  - 100 estimadores
  - Votação por maioria para classificação final
  - Random state fixo para reprodutibilidade

- **Rede Neural Artificial (RNA)**:
  - Arquitetura: 4 camadas
  - Camada de entrada: número de features
  - Camadas ocultas: 64, 32 e 16 neurônios
  - Camada de saída: 1 neurônio (sigmoid)
  - Dropout para prevenção de overfitting
  - Early stopping para otimização do treinamento

#### 5. Resultados e Comparação entre Modelos
Abaixo está uma análise comparativa entre os modelos com base nas métricas de avaliação:

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|--------|----------|----------|---------|-----------|----------|
| Random Forest | 0.6286 | 0.6842 | 0.6500 | 0.6667 | 0.6617 |
| RNA | 0.6286 | 0.6842 | 0.6500 | 0.6667 | 0.6433 |

### Análise Comparativa:
- **Melhor Desempenho**: O Random Forest obteve um desempenho ligeiramente melhor, especialmente na métrica AUC-ROC (0.6617 vs 0.6433)
- **Interpretabilidade**: 
  - Random Forest: Permite análise de importância das features, com Glucose (22.3%), Insulin (16.1%) e Age (15.6%) como as mais relevantes
  - RNA: Modelo de "caixa preta", difícil interpretação direta
- **Tempo de Treinamento**:
  - Random Forest: Treinamento mais rápido
  - RNA: Requer mais tempo devido à complexidade da arquitetura e necessidade de múltiplas épocas

### Desafios Encontrados:
- **Random Forest**:
  - Ajuste do número de estimadores
  - Balanceamento entre profundidade das árvores e overfitting
  - Interpretação do conjunto de árvores

- **Rede Neural**:
  - Definição da arquitetura ideal
  - Ajuste dos hiperparâmetros (learning rate, batch size)
  - Prevenção de overfitting
  - Sensibilidade à inicialização dos pesos

### Conclusão Final e Aprendizados:
- **Random Forest**: 
  - Apresentou o melhor desempenho geral
  - Oferece boa interpretabilidade através da análise de importância das features
  - Mais rápido para treinar e fazer previsões
  - Melhor AUC-ROC, indicando melhor capacidade de discriminação

- **Rede Neural**: 
  - Desempenho similar ao Random Forest em várias métricas
  - Requer mais tempo de treinamento
  - Difícil interpretação dos resultados
  - Sensível à inicialização e hiperparâmetros

### Aprendizados do Grupo:
Durante a execução deste trabalho, pudemos entender melhor:
- A importância da preparação adequada dos dados
- O impacto do feature engineering no desempenho dos modelos
- A necessidade de balancear complexidade e interpretabilidade
- A importância de múltiplas métricas de avaliação
- O trade-off entre desempenho e tempo de treinamento
- A necessidade de validação cruzada e técnicas de prevenção de overfitting
