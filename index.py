# pip install numpy matplotlib pandas seaborn scikit-learn keras tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

def load_dataframe(): 
	df = pd.read_csv("data/diabetes.csv")
	return df

def save_output(df, filename):
    if not os.path.exists("output"):
        os.makedirs("output")
    df.to_csv(f"output/{filename}.csv", index=False)
    print(f"Arquivo salvo em output/{filename}.csv")

def analyze_distributions(df):
	"""Analyze and plot distributions of all variables"""
	print("\n=== Análise de Distribuições ===")

	# Configurar o estilo dos gráficos
	plt.style.use('seaborn-v0_8')

	# Criar subplots para cada variável
	fig, axes = plt.subplots(3, 3, figsize=(15, 12))
	fig.suptitle('Distribuição das Variáveis', fontsize=16)

	# Flatten axes para facilitar o loop
	axes = axes.ravel()

	# Plotar histograma e KDE para cada variável
	for idx, column in enumerate(df.columns):
		sns.histplot(data=df, x=column, kde=True, ax=axes[idx])
		axes[idx].set_title(f'Distribuição de {column}')
		axes[idx].set_xlabel(column)
		axes[idx].set_ylabel('Frequência')

	plt.tight_layout()
	if not os.path.exists("output"):
		os.makedirs("output")
	plt.savefig('output/distributions.png')
	plt.close()

def analyze_correlations(df):
	"""Analyze and plot correlations between variables"""
	print("\n=== Análise de Correlações ===")

	# Calcular matriz de correlação
	corr_matrix = df.corr()

	# Plotar mapa de calor
	plt.figure(figsize=(12, 8))
	sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
	plt.title('Matriz de Correlação')
	plt.tight_layout()
	if not os.path.exists("output"):
		os.makedirs("output")
	plt.savefig('output/correlations.png')
	plt.close()

	# Mostrar correlações mais fortes
	print("\nCorrelações mais fortes:")
	corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
	print(corr_pairs[corr_pairs != 1.0].head(10))

def detect_outliers(df):
	"""Detect and analyze outliers in the dataset"""
	print("\n=== Análise de Outliers ===")

	# Configurar o estilo dos gráficos
	plt.style.use('seaborn-v0_8')

	# Criar boxplots para cada variável
	fig, axes = plt.subplots(3, 3, figsize=(15, 12))
	fig.suptitle('Boxplots para Detecção de Outliers', fontsize=16)

	# Flatten axes para facilitar o loop
	axes = axes.ravel()

	# Plotar boxplot para cada variável
	for idx, column in enumerate(df.columns):
		sns.boxplot(data=df, y=column, ax=axes[idx])
		axes[idx].set_title(f'Boxplot de {column}')

	plt.tight_layout()
	if not os.path.exists("output"):
		os.makedirs("output")
	plt.savefig('output/outliers.png')
	plt.close()

	# Análise estatística de outliers usando IQR
	print("\nEstatísticas de Outliers (método IQR):")
	for column in df.columns:
		Q1 = df[column].quantile(0.25)
		Q3 = df[column].quantile(0.75)
		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR

		outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
		if len(outliers) > 0:
			print(f"\n{column}:")
			print(f"Número de outliers: {len(outliers)}")
			print(f"Limites: [{lower_bound:.2f}, {upper_bound:.2f}]")
			print(f"Valores dos outliers: {outliers[column].values}")

def summary_statistics(df):
	"""Generate summary statistics for the dataset"""
	print("\n=== Estatísticas Descritivas ===")
	print("\nEstatísticas básicas:")
	print(df.describe())

	print("\nInformações do dataset:")
	print(df.info())

	print("\nValores nulos por coluna:")
	print(df.isnull().sum())

def remove_outliers_iqr(df, columns):
	df_clean = df.copy()
	for column in columns:
		Q1 = df_clean[column].quantile(0.25)
		Q3 = df_clean[column].quantile(0.75)
		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR
		df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
	return df_clean

def prepare_data(df):
	print("\n=== Limpeza e Preparação dos Dados ===")
	df_clean = df.copy()

	# 1. Tratar valores ausentes ou inválidos
	cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
	for col in cols_with_zero_invalid:
		num_zeros = (df_clean[col] == 0).sum()
		if num_zeros > 0:
			print(f"Coluna {col}: {num_zeros} valores zero substituídos por NaN")
			df_clean[col] = df_clean[col].replace(0, np.nan)

	# 2. Remove linhas com valores ausentes
	df_clean = df_clean.dropna()

	# 3. Normaliza variáveis numéricas (exceto Outcome)
	features = df_clean.columns.drop('Outcome')
	scaler = StandardScaler()
	df_clean[features] = scaler.fit_transform(df_clean[features])
	print("Variáveis numéricas normalizadas (StandardScaler).")

	# 4. Remove outliers
	df_clean = remove_outliers_iqr(df_clean, features)
	print("Outliers removidos usando método IQR.")

	return df_clean

def feature_engineering(df):
	"""Realiza seleção e engenharia de características"""
	print("\n=== Engenharia de Características ===")
	df_eng = df.copy()

	# 1. Análise de correlações com variáveis numéricas
	print("\nCorrelação com a variável alvo (Outcome):")
	correlations = df_eng.corr()['Outcome'].sort_values(ascending=False)
	print(correlations)

	threshold = 0.1 # Correlação mínima para manter a feature
	selected_features = correlations[abs(correlations) > threshold].index.tolist()
	selected_features.remove('Outcome') # Remove a variável alvo da lista

	print(f"\nFeatures selecionadas (correlação > {threshold}):")
	print(selected_features)

	# 2. Criar features de interação
	if all(col in df_eng.columns for col in ['BMI', 'Age']):
		df_eng['BMI_Age'] = df_eng['BMI'] * df_eng['Age']
		print("\nFeature de interação criada: BMI_Age")

	if all(col in df_eng.columns for col in ['Glucose', 'Insulin']):
		df_eng['Glucose_Insulin_Ratio'] = df_eng['Glucose'] / df_eng['Insulin']
		print("\nFeature de interação criada: Glucose_Insulin_Ratio")

	# 3. Criar variáveis categóricas
	df_eng['BMI_Category'] = pd.cut(df_eng['BMI'], 
		bins=[0, 18.5, 25, 30, 100],
		labels=['abaixo do peso', 'normal', 'sobrepeso', 'obeso'])

	df_eng['Age_Group'] = pd.cut(df_eng['Age'],
		bins=[0, 30, 45, 60, 100],
		labels=['jovem', 'adulto', 'senhor', 'idoso'])

	df_eng['Glucose_Level'] = pd.cut(df_eng['Glucose'],
		bins=[0, 100, 125, 300],
		labels=['normal', 'prediabetes', 'diabetes'])

	# 4. One-hot encoding para variáveis categóricas
	categorical_cols = ['BMI_Category', 'Age_Group', 'Glucose_Level']
	df_eng = pd.get_dummies(df_eng, columns=categorical_cols, drop_first=True)

	print("\nNovas features criadas:")
	print(df_eng.columns.tolist())

	return df_eng

def balance_classes(df):
	"""Analisa e balanceia as classes do dataset usando amostragem aleatória"""
	print("\n=== Análise e Balanceamento de Classes ===")

	# Separar features e target
	X = df.drop('Outcome', axis=1)
	y = df['Outcome']

	# Mostrar distribuição original das classes
	print("\nDistribuição original das classes:")
	class_0 = df[df['Outcome'] == 0]
	class_1 = df[df['Outcome'] == 1]

	print(f"Classe 0 (Sem diabetes): {len(class_0)} ({len(class_0)/len(df)*100:.1f}%)")
	print(f"Classe 1 (Com diabetes): {len(class_1)} ({len(class_1)/len(df)*100:.1f}%)")

	# Aplicar balanceamento apenas se houver desbalanceamento significativo
	if len(class_0) / len(class_1) > 1.5 or len(class_1) / len(class_0) > 1.5:
		print("\nAplicando balanceamento de classes...")

		# Determinar qual classe é a minoritária
		if len(class_0) < len(class_1):
			minority_class = class_0
			majority_class = class_1
		else:
			minority_class = class_1
			majority_class = class_0

		# Amostrar aleatoriamente da classe majoritária para igualar à minoritária
		majority_sampled = majority_class.sample(n=len(minority_class), random_state=42)

		# Combinar as classes balanceadas
		df_balanced = pd.concat([minority_class, majority_sampled])
		df_balanced = df_balanced.sample(frac=1, random_state=42) # Embaralhar os dados

		# Mostrar nova distribuição
		new_class_0 = df_balanced[df_balanced['Outcome'] == 0]
		new_class_1 = df_balanced[df_balanced['Outcome'] == 1]

		print("\nNova distribuição após balanceamento:")
		print(f"Classe 0 (Sem diabetes): {len(new_class_0)} ({len(new_class_0)/len(df_balanced)*100:.1f}%)")
		print(f"Classe 1 (Com diabetes): {len(new_class_1)} ({len(new_class_1)/len(df_balanced)*100:.1f}%)")

		return df_balanced
	else:
		print("\nAs classes estão relativamente balanceadas. Balanceamento não será aplicado.")
		return df

def main():
	# Carregar dados
	df = load_dataframe()

	# Realizar análise exploratória
	summary_statistics(df)
	analyze_distributions(df)
	analyze_correlations(df)
	detect_outliers(df)

	# Limpeza e preparação dos dados
	df_prepared = prepare_data(df)

	# Engenharia de características
	df_engineered = feature_engineering(df_prepared)

	# Balanceamento de classes quando há necessidade
	df_balanced = balance_classes(df_engineered)

	print("\nDataset final:")
	print(df_balanced.head())
	print("\nShape do dataset:", df_balanced.head())

if __name__ == "__main__":
	main()
