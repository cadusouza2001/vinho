# Camadas da biblioteca Keras para montar nossa rede neural
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Cria uma MLP com duas camadas ocultas
def create_model(input_dim, num_classes=3, dropout_rate=0.2):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),  # primeira camada densa com ReLU
        Dropout(dropout_rate),                              # Dropout para reduzir overfitting
        Dense(32, activation='relu'),                       # segunda camada densa
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')            # saída softmax para classificação multiclasse
    ])
    # Compilação define otimizador, função de perda e métrica de avaliação
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
