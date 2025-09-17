# models/trainer.py
import os
import tensorflow as tf

def train_model(model, x_train, x_val=None, epochs=10, batch_size=32, save_path=None):
    """
    Train the model. x_val is optional. If provided, validation_data is used.
    Returns: history
    """
    callbacks = []
    # EarlyStopping to avoid long runs:
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(monitor="val_loss" if x_val is not None else "loss",
                                   patience=7, restore_best_weights=True, verbose=1))

    if x_val is not None:
        history = model.fit(
            x_train, x_train,
            validation_data=(x_val, x_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
    # Optionally save model after training (if save_path provided)
    if save_path:
        save_model(model, save_path)
    return history

def save_model(model, path="results/models/cnn_lstm_autoencoder.keras"):
    """Save model in .keras format (preferred)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not path.endswith(".keras"):
        path = path.rsplit(".", 1)[0] + ".keras"
    model.save(path)
    print(f"[INFO] Model saved at {path}")

def load_model(path="results/models/cnn_lstm_autoencoder.keras"):
    """Load model (expects .keras)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    model = tf.keras.models.load_model(path)
    print(f"[INFO] Loaded model from {path}")
    return model
