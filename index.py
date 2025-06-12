# pip install numpy matplotlib pandas seaborn scikit-learn keras tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

def load_dataframe(): 
	"""
	Carrega o dataset de diabetes do arquivo CSV.
	Retorna:
		DataFrame pandas contendo os dados do dataset
	"""
	df = pd.read_csv("data/diabetes.csv")
	return df

def save_output(df, filename):
	"""
	Salva um DataFrame em um arquivo CSV na pasta output.
	Args:
		df: DataFrame a ser salvo
		filename: Nome do arquivo (sem extensão)
	"""
	if not os.path.exists("output"):
		os.makedirs("output")
	df.to_csv(f"output/{filename}.csv", index=False)
	print(f"Arquivo salvo em output/{filename}.csv")

def analyze_distributions(df):
	"""
	Analisa e plota as distribuições de todas as variáveis do dataset.
	Cria histogramas com KDE (Kernel Density Estimation) para cada variável.
	Args:
		df: DataFrame contendo os dados
	"""
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
	"""
	Analisa e plota as correlações entre as variáveis do dataset.
	Cria um mapa de calor (heatmap) com as correlações e mostra as correlações mais fortes.
	Args:
		df: DataFrame contendo os dados
	"""
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
	"""
	Detecta e analisa outliers no dataset usando o método IQR (Interquartile Range).
	Cria boxplots para cada variável e mostra estatísticas dos outliers.
	Args:
		df: DataFrame contendo os dados
	"""
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
	"""
	Gera estatísticas descritivas do dataset.
	Mostra estatísticas básicas, informações do dataset e contagem de valores nulos.
	Args:
		df: DataFrame contendo os dados
	"""
	print("\n=== Estatísticas Descritivas ===")
	print("\nEstatísticas básicas:")
	print(df.describe())

	print("\nInformações do dataset:")
	print(df.info())

	print("\nValores nulos por coluna:")
	print(df.isnull().sum())

def remove_outliers_iqr(df, columns):
	"""
	Remove outliers do dataset usando o método IQR.
	Args:
		df: DataFrame contendo os dados
		columns: Lista de colunas para remover outliers
	Returns:
		DataFrame com outliers removidos
	"""
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
	"""
	Prepara e limpa os dados para modelagem.
	Realiza as seguintes etapas:
	1. Trata valores zero inválidos
	2. Remove linhas com valores ausentes
	3. Normaliza variáveis numéricas
	4. Remove outliers
	Args:
		df: DataFrame contendo os dados
	Returns:
		DataFrame limpo e preparado
	"""
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
	"""
	Realiza engenharia de características no dataset.
	Inclui:
	1. Seleção de features baseada em correlação
	2. Criação de features de interação
	3. Categorização de variáveis
	4. One-hot encoding
	Args:
		df: DataFrame contendo os dados
	Returns:
		DataFrame com novas features
	"""
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
	"""
	Balanceia as classes do dataset usando undersampling.
	Args:
		df: DataFrame contendo os dados
	Returns:
		DataFrame com classes balanceadas
	"""
	print("\n=== Análise e Balanceamento de Classes ===")
	
	# Mostrar distribuição original
	class_counts = df['Outcome'].value_counts()
	print("\nDistribuição original das classes:")
	print(f"Classe 0 (Sem diabetes): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
	print(f"Classe 1 (Com diabetes): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")
	
	# Aplicar undersampling
	print("\nAplicando balanceamento de classes...")
	min_class = df[df['Outcome'] == df['Outcome'].value_counts().idxmin()]
	maj_class = df[df['Outcome'] == df['Outcome'].value_counts().idxmax()]
	
	# Amostrar aleatoriamente da classe majoritária
	maj_class_sampled = maj_class.sample(n=len(min_class), random_state=42)
	
	# Combinar as classes
	df_balanced = pd.concat([min_class, maj_class_sampled])
	
	# Mostrar nova distribuição
	new_class_counts = df_balanced['Outcome'].value_counts()
	print("\nNova distribuição após balanceamento:")
	print(f"Classe 0 (Sem diabetes): {new_class_counts[0]} ({new_class_counts[0]/len(df_balanced)*100:.1f}%)")
	print(f"Classe 1 (Com diabetes): {new_class_counts[1]} ({new_class_counts[1]/len(df_balanced)*100:.1f}%)")
	
	return df_balanced

def main():
	"""
	Função principal que executa todo o pipeline de análise e preparação dos dados.
	"""
	# Carregar dados
	df = load_dataframe()
	
	# Análise exploratória
	summary_statistics(df)
	analyze_distributions(df)
	analyze_correlations(df)
	detect_outliers(df)
	
	# Preparação dos dados
	df_clean = prepare_data(df)
	df_eng = feature_engineering(df_clean)
	df_balanced = balance_classes(df_eng)
	
	# Salvar resultados
	save_output(df_balanced, "processed_data")
	
	return df_balanced

if __name__ == "__main__":
	main()
