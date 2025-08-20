import os
import librosa
import numpy as np
import scipy.fftpack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense, Lambda
import subprocess

SR = 16000
N_MFCC = 13
MFCC_HOP = 512
AUDIO_DURATION = 1  # seconds
DATA_DIR = "16000_pcm_speeches"
MODEL_NAME = "dvector_model"

def load_audio_file(file_path, sr=SR):
    y, _ = librosa.load(file_path, sr=sr)
    if len(y) < sr * AUDIO_DURATION:
        y = np.pad(y, (0, sr * AUDIO_DURATION - len(y)))
    else:
        y = y[:int(sr * AUDIO_DURATION)]
    return y


def compute_mfcc_arduino_style(file_path):
    audio_data = load_audio_file(file_path)
    y = audio_data.astype(np.float32)
    for i in range(len(y)-1, 0, -1):
        y[i] -= 0.97 * y[i-1]


    pad_len = 256  # n_fft//2 = 512//2
    padded_y = np.pad(y, (pad_len, pad_len), mode='reflect')

    n_fft = 512
    hop_length = 512
    win_length = 512
    n_frames = 32

    # Precomputed coefficients
    hamming = np.hamming(win_length)
    mel_f = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=40,
                                htk=True, norm=1)  # norm=1 for HTK style
    dct_m = scipy.fft.dct(np.eye(40), type=2, axis=0, norm='ortho')[:13]

    # Process frames
    mfccs = np.zeros((n_frames, 13))
    for i in range(n_frames):
        # Extract frame
        start = i * hop_length
        frame = padded_y[start:start+win_length].copy()

        # Apply window
        frame *= hamming

        # FFT (using numpy with same scaling as CMSIS)
        spectrum = scipy.fft.rfft(frame, n=n_fft)
        power = (np.abs(spectrum) ** 2)  # No normalization

        # Apply mel filterbank with HTK normalization
        mel_energy = np.dot(mel_f, power)
        mel_energy = np.log(np.maximum(mel_energy, 1e-10))

        # Apply DCT with ortho normalization
        mfccs[i] = np.dot(dct_m, mel_energy)


    return mfccs.astype(np.float32)

def prepare_dataset_from_folders(root_dir):
    X = []
    y = []
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            if not file.endswith(".wav"):
                continue
            filepath = os.path.join(label_dir, file)
            mfcc = compute_mfcc_arduino_style(filepath)
            if mfcc.shape[0] < 32:
                continue
            mfcc = mfcc[:32, :]  # crop/pad to fixed shape
            X.append(mfcc)
            y.append(label)
    return np.array(X), np.array(y)


def build_dvector_model(input_shape=(32, 13, 1), embedding_dim=256):
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # d-vector output
    d_vector = output_layer = Dense(embedding_dim, activation=None, name="d_vector")(x)

    #norm_dvector = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(d_vector)

    model = Model(inputs=inputs, outputs= output_layer, name="dvector_model")
    return model


print("[INFO] Loading and processing dataset...")
print(DATA_DIR)
X, y = prepare_dataset_from_folders(DATA_DIR)

X = X[..., np.newaxis]  # Add channel dim
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

print(f"[INFO] {len(X_train)} training samples, {len(X_test)} testing samples, {num_classes} labels: {le.classes_}")

# Model with classification head
base_model = build_dvector_model()
x = Dense(num_classes, activation='softmax')(base_model.output)
clf_model = Model(inputs=base_model.input, outputs=x)

clf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
clf_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=100)


#extract mel filterbank -------------------------------------------------------------------- After training
mel_filters = librosa.filters.mel(sr=SR, n_fft=512, n_mels=40, htk=True, norm=1)
mel_filters = mel_filters[:, :512//2 + 1]  # shape: (n_mels, 257)

# Compute DCT matrix (type-II DCT, orthogonal)
dct_matrix = scipy.fftpack.dct(np.eye(40), type=2, norm='ortho', axis=0)[:13]

# Export as C arrays
def format_array_c(name, array):
    lines = []
    shape = array.shape
    flat = array.flatten()
    lines.append(f"float {name}[{shape[0]}][{shape[1]}] = {{")
    for i in range(shape[0]):
        row = ', '.join(f"{array[i, j]:.6f}f" for j in range(shape[1]))
        lines.append(f"  {{ {row} }},")
    lines.append("};\n")
    return '\n'.join(lines)

# Save to file
with open("mfcc_matrices.h", "w") as f:
    f.write("// Auto-generated MFCC matrices\n\n")
    f.write(format_array_c("melFilterBank", mel_filters))
    f.write(format_array_c("dctMatrix", dct_matrix))

print("[✅] MFCC matrices exported to mfcc_matrices.h")

def mfcc_arduino(y):
  for i in range(len(y)-1, 0, -1):
      y[i] -= 0.97 * y[i-1]

  pad_len = 256  # n_fft//2 = 512//2
  padded_y = np.pad(y, (pad_len, pad_len), mode='reflect')

  n_fft = 512
  hop_length = 512
  win_length = 512
  n_frames = 32

  hamming = np.hamming(win_length)
  mel_f = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=40,
                              htk=True, norm=1)  #  norm=1 for HTK style
  dct_m = scipy.fft.dct(np.eye(40), type=2, axis=0, norm='ortho')[:13]

  mfccs = np.zeros((n_frames, 13))
  for i in range(n_frames):

      start = i * hop_length
      frame = padded_y[start:start+win_length].copy()

      frame *= hamming

      spectrum = scipy.fft.rfft(frame, n=n_fft)
      power = (np.abs(spectrum) ** 2)  # No normalization

      mel_energy = np.dot(mel_f, power)
      mel_energy = np.log(np.maximum(mel_energy, 1e-10))

      mfccs[i] = np.dot(dct_m, mel_energy)
  return mfccs.astype(np.float32)


def representative_dataset():
    files = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for f in filenames:
            if f.endswith(".wav"):
                files.append(os.path.join(root, f))

    for fpath in files[:100]:  # use up to 100 samples
        mfcc=compute_mfcc_arduino_style(fpath)

        mfcc = np.expand_dims(mfcc, axis=-1)  # (32, 13, 1)
        mfcc = np.expand_dims(mfcc, axis=0)   # (1, 32, 13, 1)
        yield [mfcc.astype(np.float32)]

print("[INFO] Exporting base embedding model for Arduino...")

# Strip classification head
d_vector_model = Model(inputs=base_model.input, outputs=base_model.get_layer("d_vector").output)

d_vector_output = d_vector_model.output
dummy = tf.keras.layers.Activation('linear', name="quantized_output")(d_vector_output)

quant_model = Model(inputs=d_vector_model.input, outputs=dummy)

# Quantize & convert
converter = tf.lite.TFLiteConverter.from_keras_model(d_vector_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(f"{MODEL_NAME}.tflite", "wb") as f:
    f.write(tflite_model)

# Convert to Arduino .h file
print("[INFO] Converting to Arduino C++ array using xxd...")
subprocess.run(f"xxd -i {MODEL_NAME}.tflite > {MODEL_NAME}_model.h", shell=True)

print(f"[✅] Done! You can now use `{MODEL_NAME}_model.cc` in your Arduino sketch.")
