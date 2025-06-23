import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(red_path='winequality-red.csv', white_path='winequality-white.csv'):
    red = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')
    data = pd.concat([red, white], ignore_index=True)
    return data


def transform_quality(df):
    df = df.copy()
    df['quality'] = df['quality'].apply(lambda q: 0 if q <= 5 else (1 if q == 6 else 2))
    return df


def preprocess(df):
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_npz(X_train, X_test, y_train, y_test, path='preprocessed_data.npz'):
    np.savez_compressed(path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


if __name__ == '__main__':
    data = load_data()
    data = transform_quality(data)
    X, y, scaler = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    save_npz(X_train, X_test, y_train, y_test)
    print('Preprocessing completed.')
