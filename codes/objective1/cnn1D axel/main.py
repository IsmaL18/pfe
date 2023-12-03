import os
import random
import librosa
import soundfile as sf

import pandas as pd
import os
import shutil
import random



# Chemins des fichiers
source_directory = "C:\\Users\\axlco\\Downloads\\common_voice\\cv-corpus-14.0-delta-2023-06-23\\en\\clips"
tsv_file_path = "C:\\Users\\axlco\\Downloads\\common_voice\\cv-corpus-14.0-delta-2023-06-23\\en\\validated.tsv"
output_directory = "C:\\Users\\axlco\\OneDrive - ESME\\ESME Sudria\\5 ème année\\Projet\\sample"

"""

# Lire le fichier TSV
df = pd.read_csv(tsv_file_path, sep='\t')

# Catégories d'âge
age_categories = ['twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties']

# Filtrer les fichiers audio correspondant aux catégories d'âge
filtered_files = df[df['age'].isin(age_categories)]['path'].tolist()

# Prendre 1000 fichiers audio au hasard
selected_files = random.sample(filtered_files, min(1000, len(filtered_files)))

# Déplacer les fichiers vers le répertoire de sortie
for file in selected_files:
    source_path = os.path.join(source_directory, file)
    destination_path = os.path.join(output_directory, os.path.basename(file))
    shutil.copyfile(source_path, destination_path)

print(f"{len(selected_files)} fichiers ont été déplacés avec succès vers {output_directory}.")
"""

import pandas as pd
import os

# Chemins des fichiers
tsv_file_path = "C:\\Users\\axlco\\Downloads\\common_voice\\cv-corpus-14.0-delta-2023-06-23\\en\\validated.tsv"
sample_directory = "C:\\Users\\axlco\\OneDrive - ESME\\ESME Sudria\\5 ème année\\Projet\\sample"

# Lire le fichier TSV
df = pd.read_csv(tsv_file_path, sep='\t')

# Mapping des libellés d'âge en nombres
age_mapping = {'twenties': 20, 'thirties': 30, 'fourties': 40, 'fifties': 50, 'sixties': 60, 'seventies': 70, 'eighties': 80, 'nineties': 90}

# Appliquer la conversion sur la colonne 'age' du DataFrame
df['age'] = df['age'].map(age_mapping)

# Afficher les âges après la conversion
print(df['age'])

# Liste pour stocker les âges des fichiers
ages = []

# Parcourir les fichiers dans le répertoire "sample"
for file_name in os.listdir(sample_directory):
    # Construire le chemin complet du fichier
    file_path = os.path.join(sample_directory, file_name)

    # Extraire le chemin relatif du fichier par rapport au répertoire "clips"
    relative_path = os.path.relpath(file_path, sample_directory)

    # Rechercher l'âge correspondant dans le fichier TSV
    age = df.loc[df['path'] == relative_path, 'age'].values

    # Ajouter l'âge à la liste
    if len(age) > 0:
        ages.append((file_name, age[0]))

# Afficher les âges
for file_name, age in ages:
    print(f"Le fichier {file_name} a l'âge : {age}")

import pandas as pd
import os

# Chemin du répertoire contenant les fichiers audio
output_directory = "C:\\Users\\axlco\\OneDrive - ESME\\ESME Sudria\\5 ème année\\Projet\\sample"

# Créer une liste pour stocker les noms de fichiers et les âges
data = {'File Name': [], 'Age': []}

# Parcourir les fichiers dans le répertoire
for file_name in os.listdir(output_directory):
    # Construire le chemin complet du fichier
    file_path = os.path.join(output_directory, file_name)

    # Rechercher l'âge correspondant dans la liste des âges
    age = df.loc[df['path'] == file_name, 'age'].values

    # Ajouter le nom du fichier et l'âge à la liste
    if len(age) > 0:
        data['File Name'].append(file_name)
        data['Age'].append(age[0])

# Créer le DataFrame
df_result = pd.DataFrame(data)

# Afficher le DataFrame
print(df_result)


import librosa
import matplotlib.pyplot as plt
import numpy as np

# Fonction pour charger le fichier audio et obtenir le signal sous forme de vecteur 1D
def load_audio(file_path):
    # Chargement du fichier audio
    y, sr = librosa.load(file_path, sr=None)  # Ajout de sr=None pour obtenir le taux d'échantillonnage
    return y, sr

# Exemple d'utilisation de la fonction
file_path = "C:\\Users\\axlco\\OneDrive - ESME\\ESME Sudria\\5 ème année\\Projet\\sample\\common_voice_en_37285727.mp3"
audio_signal, sr = load_audio(file_path)

# Affichage du signal audio
plt.plot(np.arange(len(audio_signal)) / sr, audio_signal)
plt.title('Signal Audio 1D')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.show()

def extract_features(file_path):
    return load_audio(file_path)

X = []
y = []

# Longueur fixe pour les vecteurs 1D
fixed_length = 10000  # Choisissez une valeur appropriée
# Utiliser les valeurs converties pour former l'ensemble d'entraînement
for file_name in os.listdir(output_directory):
    file_path = os.path.join(output_directory, file_name)
    audio_signal, _ = extract_features(file_path)

    if len(audio_signal) < fixed_length:
        audio_signal = np.pad(audio_signal, (0, fixed_length - len(audio_signal)))
    elif len(audio_signal) > fixed_length:
        audio_signal = audio_signal[:fixed_length]

    X.append(audio_signal)

    # Rechercher l'âge correspondant dans le DataFrame et convertir en nombre
    age_label = df.loc[df['path'] == file_name, 'age'].values
    if len(age_label) > 0:
        y.append(int(age_label[0]))

X = np.array(X)
y = np.array(y)


print(X)
print(y)

print(X.shape)
print(y.shape)
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Supposons que les âges sont dans y
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Attribuer à chaque dizaine d'âge un numéro de classe (de 0 à 7)
num_classes = 8  # Si vous avez des âges de 20 à 90, cela donnera 8 classes (0, 1, ..., 7)
y_classes = np.round((y - 20) / 10).astype(int)




# Convertir les étiquettes en encodage one-hot avec le bon nombre de classes
y_categorical = to_categorical(y_classes, num_classes=num_classes)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train_categorical, y_test_categorical = train_test_split(X.reshape(-1, 10000, 1), y_categorical, test_size=0.2, random_state=42)

# Créer le modèle CNN 1D pour la classification d'âge
cnn_model_classification = Sequential()

# Ajouter une couche de convolution avec une fonction d'activation relu
cnn_model_classification.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(10000, 1)))

# Ajouter une couche de pooling
cnn_model_classification.add(MaxPooling1D(pool_size=2))

# Ajouter une autre couche de convolution et de pooling
cnn_model_classification.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
cnn_model_classification.add(MaxPooling1D(pool_size=2))

# Aplatir les données pour les passer à une couche Dense
cnn_model_classification.add(Flatten())

# Ajouter une couche Dense avec une fonction d'activation relu
cnn_model_classification.add(Dense(64, activation='relu'))

# Ajouter la couche de sortie avec une fonction d'activation softmax pour la classification
cnn_model_classification.add(Dense(num_classes, activation='softmax'))

# Compiler le modèle
cnn_model_classification.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Afficher un résumé du modèle
cnn_model_classification.summary()

# Entraîner le modèle
history_classification = cnn_model_classification.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_data=(X_test, y_test_categorical))

# Afficher les performances
print("Performances sur l'ensemble d'entraînement :")
train_loss, train_accuracy = cnn_model_classification.evaluate(X_train, y_train_categorical, verbose=0)
print(f"Loss: {train_loss}, Accuracy: {train_accuracy}")

print("\nPerformances sur l'ensemble de test :")
test_loss, test_accuracy = cnn_model_classification.evaluate(X_test, y_test_categorical, verbose=0)
print(f"Loss: {test_loss}, Accuracy: {test_accuracy}")
