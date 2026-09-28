import os  # used to access folders and files
import librosa  # used for audio processing
import numpy as np  # used for numerical operations


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)  
    # load the audio file

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  
    # extract 13 MFCC features

    mfcc_mean = np.mean(mfcc, axis=1)  
    # convert to fixed length (13,)

    return mfcc_mean


def load_data():
    X = []  # feature list
    y = []  # label list

    # -------- Process REAL audio files --------
    for file in os.listdir("data/real"):
        file_path = os.path.join("data/real", file)

        features = extract_features(file_path)

        X.append(features)
        y.append(0)  # 0 = REAL

    # -------- Process AI audio files --------
    for file in os.listdir("data/ai"):
        file_path = os.path.join("data/ai", file)

        features = extract_features(file_path)

        X.append(features)
        y.append(1)  # 1 = AI

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y


# Only run this part if file is executed directly
if __name__ == "__main__":
    X, y = load_data()
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)