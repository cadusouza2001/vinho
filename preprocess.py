# Importamos bibliotecas de manipulação de dados e pré-processamento
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# Função de apoio para ler os dois conjuntos de dados de vinho
# (tinto e branco) e concatená-los em um único DataFrame
def load_data(red_path='winequality-red.csv', white_path='winequality-white.csv'):
    red = pd.read_csv(red_path, sep=';')            # leitura do CSV do vinho tinto
    white = pd.read_csv(white_path, sep=';')        # leitura do CSV do vinho branco
    data = pd.concat([red, white], ignore_index=True)  # une os DataFrames
    return data


# Converte a coluna de qualidade para três classes (0, 1 ou 2)
# Baixa (<=5), média (=6) ou alta (>=7), simplificando a tarefa de classificação
def transform_quality(df):
    df = df.copy()
    df['quality'] = df['quality'].apply(lambda q: 0 if q <= 5 else (1 if q == 6 else 2))
    return df


# Separa features e rótulos, aplicando padronização (StandardScaler)
# Essa etapa faz parte do pré-processamento
def preprocess(df):
    X = df.drop('quality', axis=1).values        # matriz de atributos (features)
    y = df['quality'].values                      # vetor de classes
    scaler = StandardScaler()                    # padronização (média 0, desvio 1)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


# Divide em conjuntos de treino e teste, estratificando pelas classes
# (importante para classificação multiclasse)
def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# Salva os conjuntos de treino e teste em formato compactado
# útil para reutilização posterior
def save_npz(X_train, X_test, y_train, y_test, path='preprocessed_data.npz'):
    np.savez_compressed(path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# Execução direta para gerar arquivos pré-processados
if __name__ == '__main__':
    data = load_data()                                   # leitura dos dados originais
    data = transform_quality(data)                       # ajusta a variável alvo
    X, y, scaler = preprocess(data)                      # aplica padronização
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)  # divide dados
    save_npz(X_train, X_test, y_train, y_test)           # salva em disco
    print('Preprocessing completed.')
