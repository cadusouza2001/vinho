from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def create_model(input_dim, num_classes=3, dropout_rate=0.2):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
