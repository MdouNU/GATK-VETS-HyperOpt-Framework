import argparse
import h5py
import numpy as np
import dill
import json
import logging
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("autoencoder_anomaly_detection")
logging.basicConfig(level=logging.INFO)


def read_annotations(h5file):
    # ... (This part remains unchanged)
    return annotation_names_i, X_ni


def build_autoencoder(input_dim, encoding_dim):
    input_layer = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder


def train_autoencoder(X_ni, encoding_dim, epochs):
    scaler = MinMaxScaler()
    X_ni_scaled = scaler.fit_transform(X_ni)

    autoencoder = build_autoencoder(X_ni.shape[1], encoding_dim)
    autoencoder.fit(X_ni_scaled, X_ni_scaled, epochs=epochs, batch_size=256, shuffle=True, validation_split=0.1)

    return autoencoder, scaler


def main():
    parser = argparse.ArgumentParser()
    # ... (Argument parsing setup remains unchanged)

    args = parser.parse_args()
    annotation_names_i, X_ni = read_annotations(args.annotations_file)

    if args.hyperparameters_json_file:
        with open(args.hyperparameters_json_file) as json_file:
            hyperparameters_kwargs = json.load(json_file)
        encoding_dim = hyperparameters_kwargs.get('encoding_dim', 32)
        epochs = hyperparameters_kwargs.get('epochs', 50)
    else:
        encoding_dim = 32
        epochs = 50

    autoencoder, scaler = train_autoencoder(X_ni, encoding_dim, epochs)

    def score_samples(test_X_ni):
        test_X_ni_scaled = scaler.transform(test_X_ni)
        reconstructed = autoencoder.predict(test_X_ni_scaled)
        mse = np.mean(np.power(test_X_ni_scaled - reconstructed, 2), axis=1)
        return mse

    if args.output_prefix:
        output_model_file = f'{args.output_prefix}.autoencoder.h5'
        output_scaler_pkl_file = f'{args.output_prefix}.scaler.pkl'
        autoencoder.save(output_model_file)
        with open(output_scaler_pkl_file, 'wb') as f:
            dill.dump(scaler, f)

        logger.info(f'Autoencoder model saved to {output_model_file}')
        logger.info(f'Scaler pickled to {output_scaler_pkl_file}')


if __name__ == '__main__':
    main()

        raise

