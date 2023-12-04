import pandas as pd
import shutil
import os
import librosa
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Charger le fichier TSV dans un DataFrame en spécifiant index_col=False
file_path = '/Users/leohanifi/PycharmProjects/pythonProject/cv-corpus-5.1-2020-06-22/fr/validated.tsv'
df = pd.read_csv(file_path, delimiter='\t', header=None, index_col=False)

# Ajuster les noms des colonnes
df.columns = df.iloc[0]

# Supprimer la première ligne car elle contient les noms de colonnes inutiles maintenant
df = df[1:]

# Filtrer les lignes avec des valeurs non nulles dans la colonne 'age'
df_filtered = df[df['age'].notnull()]

# Regrouper les classes d'âge comme souhaité
age_mapping = {
    'twenties': 'twenties_thirties',
    'thirties': 'twenties_thirties',
    'fourties': 'fourties_fifties',
    'fifties': 'fourties_fifties',
    'sixties': 'sixties_seventies_eighties',
    'seventies': 'sixties_seventies_eighties',
    'eighties': 'sixties_seventies_eighties',

}

df_filtered['age'] = df_filtered['age'].replace(age_mapping)

# Afficher le DataFrame résultant
print(df_filtered[['path', 'age']])

# Définir le nombre d'échantillons par classe
sample_size = 10000
# Créer un DataFrame vide pour stocker les échantillons sélectionnés
df_filtered2 = pd.DataFrame()

# Grouper le DataFrame par la colonne 'age'
grouped_df = df_filtered.groupby('age')

# Sélectionner aléatoirement sample_size lignes de chaque groupe
for group, group_df in grouped_df:
    sampled_rows = group_df.sample(min(sample_size, len(group_df)))
    df_filtered2 = pd.concat([df_filtered2, sampled_rows])

# Réinitialiser l'index du DataFrame résultant
df_filtered2 = df_filtered2.reset_index(drop=True)

# Afficher le DataFrame résultant
print(df_filtered2[['path', 'age']])

all_paths= df_filtered2['path'].tolist()



# Définir le nouveau répertoire de destination
destination_directory = "/Users/leohanifi/PycharmProjects/pythonProject/cv-corpus-5.1-2020-06-22/fr/clips_test"

# Copier les fichiers vers le nouveau répertoire
for audio_file in all_paths:
    source_path = os.path.join("/Users/leohanifi/PycharmProjects/pythonProject/cv-corpus-5.1-2020-06-22/fr/clips", audio_file)

    # Vérifier si le fichier source existe avant de le copier
    if os.path.exists(source_path):
        destination_path = os.path.join(destination_directory, audio_file)
        shutil.copy(source_path, destination_path)
    # Aucun affichage si le fichier source n'existe pas

print(f"{len(all_paths)} fichiers ont été copiés vers {destination_directory}.")

# Calculer la répartition des âges en nombre d'occurrences
age_distribution = df_filtered2['age'].value_counts()

# Afficher la répartition des âges
print(age_distribution)

X = []  # List to store spectrograms
y = []  # List to store corresponding age labels

# Modifier la fonction extract_features
def extract_features(file_path, target_duration=3.5):
    global X, y
    # Charger le fichier audio
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')

    # Supprimer les parties silencieuses en utilisant librosa.effects.trim
    # audio, _ = librosa.effects.trim(audio)

    # Si la durée est inférieure à target_duration, ignorez le fichier
    if len(audio) < int(sr * target_duration):
        return None

    # Si la durée est supérieure à target_duration, tronquer le fichier
    if len(audio) > int(sr * target_duration):
        start_index = int((len(audio) - sr * target_duration) / 2)
        end_index = start_index + int(sr * target_duration)
        audio = audio[start_index:end_index]

    # Exemple : Spectrogramme
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Récupérer la classe d'âge associée au fichier
    file_name = os.path.basename(file_path)
    age_label = df_filtered[df_filtered['path'].str.contains(file_name)]['age'].iloc[0]

    # Ajouter le spectrogramme à la liste X
    X.append(spectrogram_db)

    # Ajouter la classe d'âge à la liste y
    y.append(age_label)


    # Remplacez cette fonction par la méthode que vous souhaitez utiliser pour extraire des caractéristiques
    return spectrogram_db

import os

# Répertoire contenant les fichiers audio
test_audio_path = '/Users/leohanifi/PycharmProjects/pythonProject/cv-corpus-5.1-2020-06-22/fr/clips_test'

# Liste des fichiers dans le répertoire
audio_files = [f for f in os.listdir(test_audio_path) if f.endswith('.mp3')]

# Parcourir la liste des fichiers et afficher le spectrogramme
for audio_file in audio_files:
    file_path = os.path.join(test_audio_path, audio_file)
    extract_features(file_path)

# Chemin du dossier contenant les fichiers audio
destination_directory = "/Users/leohanifi/PycharmProjects/pythonProject/cv-corpus-5.1-2020-06-22/fr/clips_test"

X = np.array(X)
print(len(X))
print(len(y))


y = np.array(y)

# Encoder les classes d'âge avec LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Créer un modèle CNN simple
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Former le modèle
history = model.fit(
    X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1),
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1), y_test)
)

# Évaluer le modèle sur l'ensemble de validation
y_val_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_classes = y_test

# Matrice de confusion
conf_mat = confusion_matrix(y_val_classes, y_val_pred_classes)
# Définir les classes
classes = le.classes_
# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Matrice de Confusion')
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.show()

# Tracer les courbes d'entraînement et de validation pour la précision
plt.figure(figsize=(12, 4))

# Précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Précision')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Perte')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
