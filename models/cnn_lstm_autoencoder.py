from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, RepeatVector, TimeDistributed, Dense

def build_cnn_lstm_autoencoder(window_size, n_features):
    """
    Builds a CNN-LSTM Autoencoder for time series anomaly detection.
    :param window_size: Number of timesteps per input sequence.
    :param n_features: Number of features per timestep.
    :return: Compiled Keras model.
    """
    inputs = Input(shape=(window_size, n_features))

    # CNN feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)

    # LSTM encoder
    x = LSTM(64, activation="relu", return_sequences=False)(x)

    # Repeat vector for decoder
    x = RepeatVector(window_size)(x)

    # LSTM decoder
    x = LSTM(64, activation="relu", return_sequences=True)(x)

    # TimeDistributed dense for reconstruction
    outputs = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    return model
