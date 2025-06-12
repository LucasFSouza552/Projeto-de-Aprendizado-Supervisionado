import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import os
from index import load_dataframe, prepare_data, feature_engineering, balance_classes, save_output
import seaborn as sns

def plot_roc_curves(rf_fpr, rf_tpr, rf_auc, nn_fpr, nn_tpr, nn_auc):
    """
    Plota as curvas ROC para ambos os modelos (Random Forest e Rede Neural).
    Args:
        rf_fpr, rf_tpr: Taxas de falso positivo e verdadeiro positivo do Random Forest
        rf_auc: Área sob a curva ROC do Random Forest
        nn_fpr, nn_tpr: Taxas de falso positivo e verdadeiro positivo da Rede Neural
        nn_auc: Área sob a curva ROC da Rede Neural
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})')
    plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curvas ROC - Comparação dos Modelos')
    plt.legend(loc="lower right")
    plt.savefig('output/roc_curves.png')
    plt.close()

def plot_confusion_matrices(rf_cm, nn_cm):
    """
    Plota as matrizes de confusão para ambos os modelos.
    Args:
        rf_cm: Matriz de confusão do Random Forest
        nn_cm: Matriz de confusão da Rede Neural
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Random Forest
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Matriz de Confusão - Random Forest')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Neural Network
    sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Matriz de Confusão - Neural Network')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('output/confusion_matrices.png')
    plt.close()

def train_and_evaluate_models(X, y):
    """
    Treina e avalia os modelos de machine learning.
    Args:
        X: Features do dataset
        y: Variável alvo
    Returns:
        DataFrame com os resultados comparativos dos modelos
    """
    print("\n=== Treinamento e Avaliação dos Modelos ===")
    print(f"\nShape dos dados de entrada: {X.shape}")
    print(f"Proporção das classes: {y.value_counts(normalize=True).round(3)}")
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nDados de treino:", X_train.shape)
    print("Dados de teste:", X_test.shape)
    
    # Treinar e avaliar Random Forest
    print("\n=== Random Forest ===")
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Treinar e avaliar Rede Neural
    print("\n=== Rede Neural Artificial ===")
    nn_model = train_neural_network(X_train, X_test, y_train, y_test)
    
    # Plotar curvas ROC
    plot_roc_curves(rf_model['fpr'], rf_model['tpr'], rf_model['auc'],
                   nn_model['fpr'], nn_model['tpr'], nn_model['auc'])
    
    # Plotar matrizes de confusão
    plot_confusion_matrices(rf_model['confusion_matrix'], nn_model['confusion_matrix'])
    
    # Comparar resultados
    print("\n=== Resumo Final ===")
    print("\nComparação dos modelos:")
    results_df = pd.DataFrame({
        'Random Forest': [rf_model['accuracy'], rf_model['precision'], rf_model['recall'], 
                         rf_model['f1'], rf_model['auc']],
        'Neural Network': [nn_model['accuracy'], nn_model['precision'], nn_model['recall'], 
                          nn_model['f1'], nn_model['auc']]
    }, index=['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    print(results_df)
    
    # Identificar melhor modelo por métrica
    print("\nMelhor modelo por métrica:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_model = results_df.loc[metric].idxmax()
        print(f"{metric.capitalize()}: {best_model}")
    
    return results_df

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Treina e avalia o modelo Random Forest.
    Args:
        X_train, X_test: Features de treino e teste
        y_train, y_test: Variável alvo de treino e teste
    Returns:
        Dicionário com resultados do modelo
    """
    print("\nParâmetros do modelo:")
    print("- Número de árvores: 100")
    print("- Random state: 42")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Fazer previsões
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    results = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
        'confusion_matrix': confusion_matrix(y_test, rf_pred)
    }
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, rf_pred_proba)
    results['fpr'] = fpr
    results['tpr'] = tpr
    results['auc'] = auc(fpr, tpr)
    
    # Imprimir resultados do Random Forest
    print("\nResultados do Random Forest:")
    print(f"Acurácia: {results['accuracy']:.4f}")
    print(f"Precisão: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"AUC-ROC: {results['auc']:.4f}")
    print("\nMatriz de Confusão:")
    print(results['confusion_matrix'])
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, rf_pred))
    
    # Mostrar importância das features
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 features mais importantes:")
    print(feature_importance.head())
    
    return results

def train_neural_network(X_train, X_test, y_train, y_test):
    """
    Treina e avalia o modelo de Rede Neural.
    Args:
        X_train, X_test: Features de treino e teste
        y_train, y_test: Variável alvo de treino e teste
    Returns:
        Dicionário com resultados do modelo
    """
    print("\nArquitetura do modelo:")
    print(f"- Camada de entrada: {X_train.shape[1]} neurônios")
    print("- Camada oculta 1: 64 neurônios + Dropout(0.3)")
    print("- Camada oculta 2: 32 neurônios + Dropout(0.2)")
    print("- Camada oculta 3: 16 neurônios")
    print("- Camada de saída: 1 neurônio (sigmoid)")
    
    # Criar modelo
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compilar modelo
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Early stopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss',
                                 patience=10,
                                 restore_best_weights=True)
    
    print("\nParâmetros de treinamento:")
    print("- Otimizador: Adam")
    print("- Função de perda: binary_crossentropy")
    print("- Batch size: 32")
    print("- Epochs: 100")
    print("- Early stopping: patience=10")
    
    # Treinar modelo
    history = model.fit(X_train, y_train,
                       epochs=100,
                       batch_size=32,
                       validation_split=0.2,
                       callbacks=[early_stopping],
                       verbose=1)
    
    # Fazer previsões
    nn_pred_proba = model.predict(X_test)
    nn_pred = (nn_pred_proba > 0.5).astype(int)
    
    # Calcular métricas
    results = {
        'accuracy': accuracy_score(y_test, nn_pred),
        'precision': precision_score(y_test, nn_pred),
        'recall': recall_score(y_test, nn_pred),
        'f1': f1_score(y_test, nn_pred),
        'confusion_matrix': confusion_matrix(y_test, nn_pred)
    }
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, nn_pred_proba)
    results['fpr'] = fpr
    results['tpr'] = tpr
    results['auc'] = auc(fpr, tpr)
    
    # Imprimir resultados da Rede Neural
    print("\nResultados da Rede Neural:")
    print(f"Acurácia: {results['accuracy']:.4f}")
    print(f"Precisão: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"AUC-ROC: {results['auc']:.4f}")
    print("\nMatriz de Confusão:")
    print(results['confusion_matrix'])
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, nn_pred))
    
    return results

def main():
    """
    Função principal que executa todo o pipeline de modelagem.
    """
    # Carregar e preparar dados usando as funções do index.py
    df = load_dataframe()
    df_clean = prepare_data(df)
    df_eng = feature_engineering(df_clean)
    df_balanced = balance_classes(df_eng)
    
    # Separar features e target
    X = df_balanced.drop('Outcome', axis=1)
    y = df_balanced['Outcome']
    
    # Treinar e avaliar modelos
    results = train_and_evaluate_models(X, y)
    
    # Salvar resultados
    save_output(results, "model_results")
    
    return results

if __name__ == "__main__":
    main() 