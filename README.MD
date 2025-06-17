# Projeto de Aprendizado Supervisionado

Este projeto tem como foco a aplicação e comparação de dois algoritmos clássicos de aprendizado supervisionado: **Random Forest** e **Rede Neural Artificial (RNA)**. Todo o trabalho foi desenvolvido em um único notebook Jupyter, com base em um problema real de classificação binária.

## 🎯 Objetivo

Analisar, comparar e compreender o comportamento dos modelos supervisionados em um conjunto de dados real. Mais do que alcançar a maior acurácia possível, o projeto busca interpretar os resultados, entender os desafios práticos e refletir sobre as decisões tomadas em cada etapa.

## 📁 Dataset Utilizado

O dataset escolhido foi o **Pima Indians Diabetes**, disponível gratuitamente no Kaggle:

🔗 [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Esse conjunto é composto por informações clínicas de pacientes, e a tarefa é prever se o paciente possui diabetes (classificação binária).

## 🧪 Modelos Aplicados

- **Random Forest (Scikit-learn)**
- **Rede Neural Artificial (Keras/TensorFlow)**

Ambos os modelos foram treinados com os mesmos dados de entrada para garantir uma comparação justa.

## 📋 Etapas do Projeto

O notebook (`index.ipynb`) contém as seguintes etapas:

1. **Análise Exploratória dos Dados (EDA)**  
   - Distribuições
   - Correlações
   - Detecção de outliers

2. **Pré-processamento**
   - Normalização com `StandardScaler`
   - Divisão treino/teste

3. **Treinamento dos Modelos**
   - Random Forest
   - RNA

4. **Avaliação e Métricas**
   - Acurácia
   - Precisão
   - Revocação
   - F1-score
   - Curva ROC e AUC
   - Matriz de Confusão

5. **Comparação Final dos Resultados**

## 📈 Resultados

A Random Forest se mostrou eficiente com poucos ajustes e interpretável em relação às variáveis mais importantes. Já a RNA precisou de mais atenção no pré-processamento, mas obteve bons resultados com ajuste de camadas e uso de `EarlyStopping`.

## 📦 Requisitos

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
```
---

## ▶️ Como Executar no VSCode

1. Clone o repositório:
   ```bash
   git clone https://github.com/LucasFSouza552/Projeto-de-Aprendizado-Supervisionado
   cd Projeto-de-Aprendizado-Supervisionado
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. No VSCode, abra o arquivo `index.ipynb` diretamente.  
   Certifique-se de que a extensão do Jupyter está instalada.

---

## 🧱 Estrutura e Organização do Código

Todo o projeto foi desenvolvido no notebook `index.ipynb`, estruturado com células modulares para facilitar a leitura e a reutilização de código. A seguir, estão as principais funções e blocos implementados:

### 📦 Pré-processamento e Análise
- `load_dataframe()`: carrega o dataset CSV original.
- `summary_statistics()` e `analyze_distributions()`: realizam a análise exploratória dos dados.
- `remove_outliers_iqr()` e `detect_outliers()`: aplicam a detecção e remoção de outliers via método IQR.
- `prepare_data()`: executa limpeza, normalização e remoção de ruídos.
- `feature_engineering()`: gera novas variáveis com base na relação entre atributos.
- `balance_classes()`: aplica undersampling para balancear as classes.

### 🧪 Modelagem e Avaliação
- `create_model_rna()`: define a arquitetura da RNA (Rede Neural Artificial) com Keras.
- `create_model_random_forest()`: cria e treina um modelo Random Forest com pesos balanceados.
- `evaluate_model()`: gera métricas como acurácia, precisão, recall, F1-score e curva ROC.

### 📊 Visualizações
- `plot_training_history()`: exibe a evolução da acurácia e perda da RNA.
- `plot_confusion_matrix()`: gera a matriz de confusão.
- `plot_roc_curve()`: plota as curvas ROC comparativas entre os modelos.

### 🧰 Execução Final
A execução completa do pipeline é iniciada pela função `prepare_data_pipeline()`, seguida pelas funções `run_rna_model_evaluation()` e `run_random_forest_evaluation()`, encerrando com o comparativo gráfico via `plot_roc_curve()`.

---

## 👥 Desenvolvedores

Este projeto foi desenvolvido por estudantes durante uma atividade prática de aprendizado supervisionado. Conheça os integrantes do grupo:

- [Lucas F. Souza](https://github.com/LucasFSouza552)
- [João A. Souza](https://github.com/JoaoASouzaN)
- [Breno Jhonson](https://github.com/BrenoJhonson)

Contribuímos juntos na análise, modelagem e desenvolvimento das soluções, cada um focando em partes distintas do projeto, como visualizações, engenharia de atributos e ajustes dos modelos.
