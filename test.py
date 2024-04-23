import numpy as np
import tensorflow as tf

fs = 2.4e6

def generate_wow_signal(n_samples=1024, freq=1420.40E6, drift_rate=0):
    # Generate a synthetic "Wow!" signal
    # The "Wow!" signal was a strong, narrowband signal at around 1420 MHz with a bandwidth of around 10 Hz
    # We'll generate a signal with the same frequency and bandwidth, but with a random amplitude and phase
    t = np.linspace(0, n_samples / fs, n_samples)
    f = freq + drift_rate * t
    x = np.sin(2 * np.pi * f * t)
    x += np.random.randn(n_samples) * 0.1  # Add some noise
    return x

def generate_noise_sample(n_samples=1024):
    # Generate a noise sample
    return np.random.randn(n_samples)

def create_model():
    # Create a machine learning model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Reshape((1024, 1)),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # Generate training data
    wow_signals = [generate_wow_signal(n_samples=1024) for _ in range(1000)]
    noise_samples = [generate_noise_sample(n_samples=1024) for _ in range(1000)]
    X_train = np.concatenate([wow_signals, noise_samples])
    y_train = np.concatenate([np.ones(1000), np.zeros(1000)])

    # Reshape for convolutional input
    X_train = np.reshape(X_train, (-1, 1024, 1))

    # Create and train model
    model = create_model()
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Save the trained weights
    model.save_weights('signal_classifier_weights.h5')

def test_model(model):
    # Generate test data
    wow_signals = [generate_wow_signal(n_samples=1024) for _ in range(100)]
    noise_samples = [generate_noise_sample(n_samples=1024) for _ in range(100)]
    X_test = np.concatenate([wow_signals, noise_samples])
    y_test = np.concatenate([np.ones(100), np.zeros(100)])

    # Reshape for convolutional input
    X_test = np.reshape(X_test, (-1, 1024, 1))

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = np.mean(y_pred.round() == y_test)
    print(f'Test accuracy: {accuracy:.2f}')

def main():
    # Load the machine learning model
    model = create_model()
    train_model()
    
    model.load_weights('signal_classifier_weights.h5')  # Provide the path to the trained weights

    # Test the model
    test_model(model)

if __name__ == '__main__':
    main()
