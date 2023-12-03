import pandas as pd

import numpy as np

from tqdm import tqdm

import librosa

import math

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings("ignore")





# ----------- CREATION DES VARIABLES ----------

tsv_file_path = "../cv-corpus_5.1/fr/validated.tsv"
mp3_files_folder = "../cv-corpus_5.1/fr/clips/"

valid_genders = ['male', 'female']
valid_ages = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties'] # il n'y a pas de nineties dans le dataset

num_classes_age = 8 # il n'y a pas de nineties dans le dataset

choice = 'age'

X_age = []
X_gender = []
y_gender = []
y_age = []
X_train_age = []
X_test_age = []
y_age_train = []
y_age_test = []
X_train_gender = []
X_test_gender = []
y_gender_train = []
y_gender_test = []

df_age = None
df_gender = None

threshold_age_column = 500  # age --> twenties : 84109   &   thirties : 81491   &   fourties : 51617   &   fifties : 47472   &   sixties : 17740   &   teens : 13502   &   seventies : 2329   &   eighties : 24
threshold_gender_column = 50000 # gender --> male : 272861   &   female : 43749

model_rnn_age = None
model_rnn_gender = None
history_rnn_age = None
history_rnn_gender = None

max_amp = 0
audio_time = 3 # temps en secondes d'un audio

version_age = '5'
version_gender = '1'

nbr_epochs = 200
batch_size = 32
patience = 10





# ---------- IMPORT DU DATASET ----------

def limit_number_values_in_dataset(df, column, threshold): # cette fonction permet d'empêcher qu'une valeur soit représentée trop de fois dans une colonne d'un dataset, cela va permettre qu'il n'y est pas une ou plusieurs valeurs qui soeint trop représentées dans nos données
    values_count = df[column].value_counts()
    different_values = values_count.index.tolist()

    new_df = df

    for val in different_values:
        if values_count[val] > threshold:
            filtered_df = df[df[column] == val].head(threshold)

            new_df = pd.concat([new_df[new_df[column] != val], filtered_df])

        else:
            pass

    new_df = new_df.sample(frac=1)

    return new_df

def load_dataset():
    print("\n\n\n\n---------- LOAD DATASET ----------")

    global df_age, df_gender

    if choice == 'age':
        df_age = pd.read_csv(tsv_file_path, delimiter='\t')
        df_age = df_age[df_age['age'].isin(valid_ages)]
        df_age = limit_number_values_in_dataset(df_age, 'age', threshold_age_column)

    elif choice == 'gender':
        df_gender = pd.read_csv(tsv_file_path, delimiter='\t')
        df_gender = df_gender[df_gender['gender'].isin(valid_genders)]
        df_gender = limit_number_values_in_dataset(df_gender, 'gender', threshold_gender_column)





# ---------- ASSOCIATION DES FICHIERS MP3 AVEC LEURS LABELS + PREPROCESSING 1/2 (troncage des audios) ----------

def association_file_label():
    print("\n\n\n\n---------- ASSOCIATION FILE-LABEL ----------")

    global X_gender, y_gender, X_age, y_age, max_amp

    if choice == 'gender':
        for idx in tqdm(range(len(df_gender)), desc='Association file-label ', colour='white'):
            filepath = mp3_files_folder + df_gender.iloc[idx]['path']

            audio, sr = librosa.load(filepath)

            if len(audio) < audio_time*sr:
                pass

            else:
                nbr_samples = math.floor(len(audio)/(audio_time * sr))

                step = 0

                for i in range(1, nbr_samples+1):
                    new_audio = audio[step:i*audio_time*sr:2] # pas de 2 pour réduire le nombre d'échantillons par vecteur et donc éviter d'utiliser trop de mémoire

                    if max_amp < np.max(np.abs(new_audio)):
                        max_amp = np.max(np.abs(new_audio))

                    X_gender.append(new_audio)
                    y_gender.append(df_gender.iloc[idx]['gender'])

                    step += audio_time * sr

    if choice == 'age':
        for idx in tqdm(range(len(df_age)), desc='Association file-label ', colour='white'):
            filepath = mp3_files_folder + df_age.iloc[idx]['path']

            audio, sr = librosa.load(filepath)

            if len(audio) < audio_time * sr:
                pass

            else:
                nbr_samples = math.floor(len(audio) / (audio_time * sr))

                step = 0

                for i in range(1, nbr_samples + 1):
                    new_audio = audio[step:i*audio_time*sr:2] # pas de 2 pour réduire le nombre d'échantillons par vecteur et donc éviter d'utiliser trop de mémoire

                    if max_amp < np.max(np.abs(new_audio)):
                        max_amp = np.max(np.abs(new_audio))

                    X_age.append(new_audio)
                    y_age.append(df_age.iloc[idx]['age'])

                    step += audio_time * sr





# ---------- PREPROCESSING 2/2 (normalisation) -------------

def label_encoding():
    global y_gender, y_age

    if choice == 'gender':
        for idx_label in range(len(y_gender)):
            if y_gender[idx_label] == 'male':
                y_gender[idx_label] = 0
            elif y_gender[idx_label] == 'female':
                y_gender[idx_label] = 1

    elif choice == 'age':
        new_y = []

        for i in range(len(y_age)):
            vect = []

            for j in range(len(valid_ages)):
                if y_age[i] == valid_ages[j]:
                    vect.append(1)
                else:
                    vect.append(0)

            new_y.append(vect)

        y_age = new_y

def normalize_audios():
    global X_gender, X_age

    if choice == 'age':
        for idx_audio in tqdm(range(len(X_age)), desc='Normalization of every audios ', colour='yellow'):
            X_age[idx_audio] = X_age[idx_audio] / max_amp

    elif choice == 'gender':
        for idx_audio in tqdm(range(len(X_gender)), desc='Normalization of every audios ', colour='yellow'):
            X_gender[idx_audio] = X_gender[idx_audio] / max_amp

def preprocessing():
    print("\n\n\n\n---------- PREPROCESSING ----------")

    global X_gender, y_gender, X_age, y_age, X_train_age, X_test_age, y_age_train, y_age_test, X_train_gender, X_test_gender, y_gender_train, y_gender_test

    label_encoding()
    normalize_audios()

    if choice == 'age':
        X_age = np.array(X_age)
        y_age = np.array(y_age)

        X_train_age, X_test_age, y_age_train, y_age_test = train_test_split(X_age, y_age, test_size=0.2, random_state=42)

        X_train_age = X_train_age.reshape(X_train_age.shape[0], X_train_age.shape[1], 1)

    elif choice == 'gender':
        X_gender = np.array(X_gender)
        y_gender = np.array(y_gender)

        X_train_gender, X_test_gender, y_gender_train, y_gender_test = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42)





# ---------- CREATION DES MODELE ----------

def creation_rnn_model():
    print("\n\n\n\n---------- CREATION OF THE MODEL ----------")

    global model_rnn_gender, model_rnn_age
    if choice == 'gender':
        model_rnn_gender = Sequential()
        model_rnn_gender.add(SimpleRNN(64, input_shape=(X_gender.shape[1], 1)))
        model_rnn_gender.add(Dense(1, activation='sigmoid'))

        model_rnn_gender.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif choice == 'age':
        print("\n\n\nX_train_age.shape : ", X_train_age.shape, "\n\n\n")

        model_rnn_age = Sequential()
        model_rnn_age.add(SimpleRNN(16, input_shape=(X_train_age.shape[1], X_train_age.shape[2]), return_sequences=True))
        model_rnn_age.add(SimpleRNN(32, return_sequences=True))
        model_rnn_age.add(SimpleRNN(64))
        model_rnn_age.add(Dense(num_classes_age, activation='softmax'))

        model_rnn_age.summary()

        # Modèle de base
        """model_rnn_age = Sequential()
        model_rnn_age.add(SimpleRNN(64, input_shape=(X_train_age.shape[1], 1)))
        model_rnn_age.add(Dense(num_classes_age, activation='softmax'))"""  # Utilisation de softmax pour la classification multiclasse

        # Modèle Axel
        """model_rnn_age = Sequential()
        model_rnn_age.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_age.shape[1], 1)))
        model_rnn_age.add(MaxPooling1D(pool_size=2))
        model_rnn_age.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
        model_rnn_age.add(MaxPooling1D(pool_size=2))
        model_rnn_age.add(Flatten())
        model_rnn_age.add(Dense(64, activation='relu'))
        model_rnn_age.add(Dense(num_classes_age, activation='softmax'))"""

        model_rnn_age.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model_rnn_age.summary()





# ---------- ENTRAINEMENT DU MODELE ET CREATION DES CALLBACKS -----------

def fit_model():
    print("\n\n\n\n---------- FIT THE MODEL ----------")

    global history_rnn_age, history_rnn_gender

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, mode='max', verbose=1)

    if choice == 'age':
        filename_model = 'rnn_model_' + choice + '_v' + version_age + '.h5'
        checkpoint = ModelCheckpoint(filename_model, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

        history_rnn_age = model_rnn_age.fit(X_train_age, y_age_train, epochs=nbr_epochs,
                                               batch_size=batch_size, verbose=1, validation_data=(X_test_age, y_age_test),
                                               callbacks=[checkpoint, early_stopping])
        """history_rnn_age = model_rnn_age.fit(X_train_age.reshape(X_train_age.shape[0], X_train_age.shape[1], 1), y_train_age, epochs=nbr_epochs,
                                            batch_size=batch_size, verbose=1, validation_data=(X_test_age, y_age_test),
                                            callbacks=[checkpoint, early_stopping])"""

    elif choice == 'gender':
        filename_model = 'rnn_model_' + choice + '_v' + version_gender + '.h5'
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

        history_rnn_gender = model_rnn_gender.fit(X_gender.reshape(X_gender.shape[0], X_gender.shape[1], 1), y_gender,
                                                  epochs=nbr_epochs, batch_size=batch_size, verbose=1,
                                                  validation_data=(X_test_gender, y_gender_test),
                                                  callbacks=[checkpoint, early_stopping])





# ---------- EVALUATION DU MODELE ----------

def plot_graphs(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def f1_conf_matrix(X_test, y_test):
    if choice == 'gender':
        predictions = model_rnn_gender.predict(X_test)
    elif choice == 'age':
        predictions = model_rnn_age.predict(X_test)

    if choice == 'gender':
        for idx_pred in range(len(predictions)):
            if predictions[idx_pred] < 0.5:
                predictions[idx_pred] = 0
            else:
                predictions[idx_pred] = 1

    if choice == 'age':
        predicted_classes = []
        y_true_labels = []

        for idx_pred in range(len(predictions)):
            prediction = np.array(predictions[idx_pred])
            predicted_class = np.argmax(prediction) + 1

            predicted_classes.append(predicted_class)

            for j in range(len(y_test[idx_pred])):
                if y_test[idx_pred][j] == 1:
                    true_label = j + 1
                    y_true_labels.append(true_label)
                else:
                    pass

    print(y_true_labels, "\nVS\n", predicted_classes)



    print("Matrice de confusion :")
    print(confusion_matrix(y_true_labels, predicted_classes))
    print("F1-score : ")
    print(f1_score(y_true_labels, predicted_classes, average='weighted'))


def evaluate():
    print("\n\n\n\n---------- EVALUATE THE MODEL ----------")

    if choice == 'gender':
        plot_graphs(history_rnn_gender)
        f1_conf_matrix(X_test_gender, y_gender_test)

    if choice == 'age':
        f1_conf_matrix(X_test_age, y_age_test)
        plot_graphs(history_rnn_age)





# ---------- MAIN ----------
def main_rnn1d_51():
    load_dataset()
    association_file_label()
    preprocessing()
    creation_rnn_model()
    fit_model()
    evaluate()