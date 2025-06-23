import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import preprocess
import model as nn_model


def main():
    data = preprocess.load_data()
    data = preprocess.transform_quality(data)
    X, y, scaler = preprocess.preprocess(data)
    X_train, X_test, y_train, y_test = preprocess.train_test_split_data(X, y)

    model = nn_model.create_model(X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es]
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy: {test_acc:.4f}')
    model.save('wine_quality_model.h5')


if __name__ == '__main__':
    main()
