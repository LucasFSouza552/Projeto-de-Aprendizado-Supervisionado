import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

def load_dataframe(): 
	df = pd.read_csv("data/diabetes.csv")
	return df

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
	plt.savefig('distributions.png')
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
	plt.savefig('correlations.png')
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
	plt.savefig('outliers.png')
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

	print(df_prepared.head()) 

if __name__ == "__main__":
	main()
