import logging
import numpy as np
from scipy.signal import lfilter, butter, freqz
import argparse
import socket
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import gc
import scipy.signal
import datetime
import os
import keyboard

# Buffer to hold received samples
buffer = []
# Sampling frequency in Hz
fs = 10000
# Low band LO frequency in MHz
notch_freq = 9750
# Notch width in MHz
notch_width = 30

def receive_samples_from_server(client_socket):
    """Receive samples from the server"""
    samples = client_socket.recv(1024)
    samples_array = np.frombuffer(samples, dtype=np.uint8)
    return samples_array

def remove_lnb_effect(signal, fs, notch_freq, notch_width):
    """Remove LNB (Low-Noise Block) effect from the signal."""
    signal = np.asarray(signal, dtype=np.float64)
    # Calculate the notch filter coefficient
    t = np.tan(np.pi * notch_width / fs)
    beta = (1 - t) / (1 + t)
    gamma = -np.cos(2 * np.pi * notch_freq / fs)

    # Notch filter coefficients
    b = [1, gamma * (1 + beta), beta]
    a = [1, gamma * (1 - beta), -beta]

    # Apply the notch filter to the signal
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal

def create_model():
    """Create a machine learning model"""
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

def process_samples(samples, fs, lnb_offset, low_cutoff, high_cutoff):
    """Process the received samples"""
    # Remove LNB effect
    processed_samples = remove_lnb_effect(samples, fs, lnb_offset,notch_width)

    # Apply bandpass filter
    nyq_rate = fs / 2.0
    low_cutoff = low_cutoff / nyq_rate
    high_cutoff = high_cutoff / nyq_rate
    b, a = butter(4, [low_cutoff, high_cutoff], btype='bandpass')
    processed_samples = lfilter(b, a, processed_samples)

    return processed_samples

def generate_wow_signal(n_samples=1024, freq=1420.40E6, drift_rate=10):
    """Generate a synthetic 'Wow!' signal"""
    # Generate a signal with the same frequency and bandwidth, but with a random amplitude and phase
    t = np.linspace(0, n_samples / fs, n_samples)
    f = freq + drift_rate * t
    x = np.sin(2 * np.pi * f * t)
    x += np.random.randn(n_samples) * 0.1  # Add some noise
    return x

def generate_noise_sample(n_samples=1024):
    """Generate a noise sample"""
    return np.random.randn(n_samples)

def train_model():
    """Train the machine learning model"""
    # Generate training data
    wow_signals = []
    noise_samples = []
    for _ in range(1000):
        # Generate 'Wow!' signal with random amplitude, frequency, and noise level
        amplitude = np.random.uniform(0.1, 0.5)
        freq = np.random.uniform(1420.3e6, 1420.5e6)
        noise_level = np.random.uniform(0.05, 0.2)
        wow_signal = amplitude * generate_wow_signal(n_samples=1024, freq=freq, drift_rate=10)
        noise = noise_level * generate_noise_sample(n_samples=1024)
        wow_signal += noise
        wow_signals.append(wow_signal)

        # Generate noise sample with random noise level
        noise_level = np.random.uniform(0.05, 0.2)
        noise_sample = noise_level * generate_noise_sample(n_samples=1024)
        noise_samples.append(noise_sample)

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
    """Test the machine learning model"""
    # Generate test data
    wow_signals = []
    noise_samples = []
    for _ in range(100):
        # Generate 'Wow!' signal with random amplitude, frequency, and noise level
        amplitude = np.random.uniform(0.1, 0.5)
        freq = np.random.uniform(1420.3e6, 1420.5e6)
        noise_level = np.random.uniform(0.05, 0.2)
        wow_signal = amplitude * generate_wow_signal(n_samples=1024, freq=freq, drift_rate=10)
        noise = noise_level * generate_noise_sample(n_samples=1024)
        wow_signal += noise
        wow_signals.append(wow_signal)

        # Generate noise sample with random noise level
        noise_level = np.random.uniform(0.05, 0.2)
        noise_sample = noise_level * generate_noise_sample(n_samples=1024)
        noise_samples.append(noise_sample)

    X_test = np.concatenate([wow_signals, noise_samples])
    y_test = np.concatenate([np.ones(100), np.zeros(100)])

    # Reshape for convolutional input
    X_test = np.reshape(X_test, (-1, 1024, 1))

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = np.mean(y_pred.round() == y_test)
    print(f'Test accuracy: {accuracy:.2f}')

def predict_signal(model, samples, threshold):
    """Predict if a signal is present"""
    # Reshape the input samples to the format the model expects
    reshaped_samples = samples[np.newaxis, :]

    # Make the prediction
    confidence = model.predict(reshaped_samples)

    # Return whether it's a signal or not based on the threshold
    return confidence[0][0] >= threshold

def plot_signal_strength(signal_strength, filename):
    """Plot signal strength"""
    plt.plot(signal_strength)
    plt.title('Signal Strength')
    plt.xlabel('Samples')
    plt.ylabel('Strength')
    plt.savefig(filename)
    plt.close()

def analyze_signal(samples):
    """Analyze the signal"""
    # Take FFT
    fft = np.fft.fft(samples)

    # Frequency spectrum
    freqs = np.abs(fft)

    # Calculate the frequency resolution
    freq_resolution = args.sampling_frequency / len(samples)

    # Return dominant frequency
    return np.argmax(freqs) * freq_resolution

def plot_spectrogram(signal, sample_rate, title='Spectrogram'):
    """Plot the spectrogram"""
    datime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'spectrograph_{datime}.png'
    # Set spectrogram parameters
    nperseg = 1024
    noverlap = nperseg // 2

    # Take the spectrogram
    f, t, Sxx = scipy.signal.spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # Plot the spectrogram
    fig = plt.figure()
    plt.imshow(Sxx, aspect='auto', extent=[f.min(), f.max(), 0, 100])
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_address, args.server_port))
    signal_strength = []
    threshold2 = 1024
    a = 0
    b = False
    try:
        params = {
            'start_freq': args.frequency,
            'end_freq': args.frequency,
            'single_freq': True,
            'sample_rate': args.sampling_frequency,
            'duration_seconds': args.duration
        }
        params_str = json.dumps(params)

        # Encode the JSON string and send it over the socket
        client_socket.sendall(params_str.encode())
        model = create_model()
        # Train the model
        if os.path.isfile('signal_classifier_weights.h5'):
            # Load the trained weights
            model.load_weights('signal_classifier_weights.h5')
            print('Loaded trained weights from file.')
        else:
            # Train the model and save the weights
            train_model()
            print('Trained new model and saved weights to file.')
        buffer_test = []
        start = time.time()
        while True:
            try:
                if time.time() - start > 60:
                    gc.collect()
                    start = time.time()

                samples = receive_samples_from_server(client_socket)
                buffer.extend(samples)

                # Process samples if the buffer size is larger than a threshold
                if len(buffer) >= 1024:  # Change the threshold as needed
                    low_cutoff = (1420.3e6 - notch_freq) / (args.sampling_frequency / 2.0)
                    high_cutoff = (1420.5e6 - notch_freq) / (args.sampling_frequency / 2.0)
                    processed_samples = process_samples(buffer, args.sampling_frequency, notch_freq, low_cutoff, high_cutoff)
                    strength = np.mean(np.abs(processed_samples))
                    signal_strength.append(strength)
                    threshold = 0.5

                    buffer_test.append(processed_samples)

                    if len(buffer_test) >= 1e8:
                        buffer_test.clear()

                    # Check if a signal is detected
                    if predict_signal(model, processed_samples,threshold):
                        psamples = process_samples
                        print(psamples)
                        print('predicting')
                        datime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        print("Signal detected!")
                        filenamedata = f'signal_samples_{datime}.dat'
                        with open(filenamedata, 'ab') as f:
                            np.savetxt(f, processed_samples)

                        filename = f'signal_strength_{datime}.png'
                        plot_signal_strength(signal_strength,filename)
                        a += 1
                        freq = analyze_signal(processed_samples)
                        print(f"Dominant Frequency: {freq} Hz")
                        plot_spectrogram(np.array(buffer_test), args.sampling_frequency)

                    buffer.clear()

                    if len(signal_strength) > threshold2:
                        signal_strength = []

                time.sleep(0.2)
            except KeyboardInterrupt:
                logging.info("Keyboard interrupt detected. Closing connection.")
                break
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Closing connection.")
    finally:
        client_socket.close()

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Process stream data from RTL-SDR server.')
    parser.add_argument('-a', '--server-address', type=str, default='localhost', help='Server IP address')
    parser.add_argument('-p', '--server-port', type=int, default=8888, help='Server port')
    parser.add_argument('-o', '--lnb-offset', type=float, default=9750e6, help='LNB offset frequency in Hz')
    parser.add_argument('-f', '--frequency', type=float, default=100e6, help='Center frequency in Hz')
    parser.add_argument('-g', '--gain', type=float, default='auto', help='Gain setting')
    parser.add_argument('-s', '--sampling-frequency', type=float, default=2.4e6, help='Sampling frequency in Hz')
    parser.add_argument('-e', '--end-frequency', type=float, default=100e6, help="aaa")
    parser.add_argument('-t', '--duration', type=int, default=60, help='time in seconds')
    parser.add_argument('-z', '--zed', type=int, default=0, help='create map 0 no 1 yes')
    args = parser.parse_args()
    main()
