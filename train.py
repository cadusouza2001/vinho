# Bibliotecas necessárias para treinamento
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import preprocess                   # módulo de pré-processamento
import model as nn_model            # definição da arquitetura da rede


# Função principal de treinamento da rede
def main():
    data = preprocess.load_data()                      # leitura dos arquivos CSV
    data = preprocess.transform_quality(data)          # conversão para classes 0,1,2
    X, y, scaler = preprocess.preprocess(data)         # normalização dos dados
    X_train, X_test, y_train, y_test = preprocess.train_test_split_data(X, y)  # separação treino/teste

    model = nn_model.create_model(X_train.shape[1])    # cria a MLP
    # EarlyStopping interrompe o treinamento caso a validação não melhore
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,              # parte dos dados de treino vira validação
        callbacks=[es]
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)   # avalia no conjunto de teste
    print(f'Test accuracy: {test_acc:.4f}')
    model.save('wine_quality_model.h5')               # salva o modelo treinado


# Executa treinamento quando o script é chamado diretamente
if __name__ == '__main__':
    main()
