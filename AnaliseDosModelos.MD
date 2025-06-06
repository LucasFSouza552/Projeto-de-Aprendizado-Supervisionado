# Relatorio Final - Analise de Modelos de Classificação

#### 1. Introdução ao Problema:
Este trabalho teve como objetivo aplicar e comparar diferentes algoritmos de classificação em um conjunto de dados real, visando a classificação correta das classes alvo. Tendo como prioridade analisar o desempenho dos modelos, sua interpretabilidade e eficiencia. Entender os desafios enfrentados durante todo o processo.

#### 2. Descrição do Dataset:
O dataset utilizado é Pima Indians Diabetes e ele possue as seguintes caracteristicas: 
- Aproximadamente 768 linhas
- 8 variáveis preditoras
- Variável alvo: Outcome (0 = não diabético, 1 = diabético)
- Atributos: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction e Age.

#### 3. EDA(Analise Exploratoria de Dados) e Preparação dos Dados: 
Ao fazer a analise exploratoria dos daos aplicamos as seguintes etapas, em busca da validação, tratamento e normatização dos dados.

- Verificação de valores ausentes
- Análise de distribuição das variávei
- Correlação entre atributos
- Balanceamento da variável alvo (se necessário)
- Normalização dos dados (aplicada especialmente para o KNN)
- Divisão do conjunto em treino e teste

#### 4. Descricao dos Modelos Implementados
Foram implementados e avaliados três modelos principais:

- Arvore de descisao (Dicision Tree): modelo baseado em regras de dicisao binaria que busca maximizar a pureza dos nos.
- Random Forest: Conjunto de arvores de decisao treinados de forma alratoria com a votacao por maioria.
- K-Nearest Neighbors(KNN): algoritmos baseados em distancia, que classifica uma boca amostra com base nas classes mais proximas no espaco de caracteristicas.

#### 5. Resultado e Comparacao entre Modelos
Abaixo esta uma analise comparativa entre os modelos com vase na aciracia e tempo de execucao:

| Modelo | Acuracia(%) | Tempo de treino | interpretabilidade | Desafios Encontrados |
| ----------- | ----------- | ----------- | ----------- |----------- |
| Arvore decisao | 83.3 | Muito rapido | Alta | Propenso a overfitting |
| Random Forest | 93.3 | Medio | Media | Complexidade e dificil interpretabilidade |
| KNN | 63.3 | Lento | Baixa | Sensivel a Normalizacao e ao valor do K |

### Analise Comparativa:
- **Melhor Desempenho(aciracia):** A random forest obteve o melhor desempenho com 93,3% de acuracia, superando significativamente os outros dois.
- **Interpretabilidade:** A Arvore de Decisao é a mais facil de interpretar, pois apresenta regras claras e faceis de visualizar. Já o Random Forest é mais dificil de analisar individualmente por ser um modelo de "caixa preta".
- **Facilidade e velocidade:** Mais rapido de treinar: A arvore de decisao foi mais facil e rapida de se treinar, seguida pelo Random Forest. O KNN, por ser preguiçoso()