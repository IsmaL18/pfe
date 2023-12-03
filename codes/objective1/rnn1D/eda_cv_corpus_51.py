"""import subprocess
pd_installation_command = 'pip install pandas'
lib_installation_command = 'pip install librosa'
ipython_installation_command = 'pip install IPython'
soundd_installation_command = 'pip3 install sounddevice'
tqdm_installation_command = 'pip3 install tqdm'
plt_installation_command = 'pip3 install matplotlib'
subprocess.check_call(plt_installation_command, shell=True)"""

import pandas as pd
import librosa
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



filenames_list = [["validated.tsv"], ["invalidated.tsv"], ["other.tsv"], ["train.tsv"], ["test.tsv"], ["dev.tsv"]]
files_tsv_folder_path = "../cv-corpus_5.1/fr/"
mp3_files_folder = "../cv_corpus_5.1/fr/clips/"





def load_tsv_files(print_choice=0): # print_choice=1 if you want to print the informations
    for filename in filenames_list:
        file_path = files_tsv_folder_path + filename[0]
        df = pd.read_csv(file_path, delimiter='\t')
        columns = df.columns.values.tolist()

        # Print informations about the dataframes
        if print_choice == 1:
            print("\n\n", filename[0], " : \n")
            print("filepath : ", file_path, "\n")
            print("Columns : ", columns, "\n")
            print("df.head(5) :\n", df.head(5), "\n")

            print("Example : ")
            for col in columns:
                print(col, " : ", df.iloc[1].loc[col])

        else:
            pass

        filename.append(df)





def get_wrong_values(filename, print_choice=0):
    for file in filenames_list:
        if file[0] == filename:
            df = file[1]

    # Print informations about the dataframe
    if print_choice == 1:
        print("Length dataset : ", len(df), "\n\n")
        print("Data types :\n ", df.dtypes, "\n\n")
        print("Nulls : \n", df.isnull().sum(), "\n\n")
        print("Age labels : \n", df['age'].value_counts(), "\n\n")
        print("Gender labels : \n", df['gender'].value_counts(), "\n\n")

    # Print wrong age and gender values
    wrong_values_age = []
    wrong_values_gender = []
    for i in range(len(df)):
        if not str(df['age'].iloc[i]) in ['thirties', 'twenties', 'fifties', 'fourties', 'teens', 'sixties', 'seventies', 'eighties', 'nineties']:
            wrong_values_age.append(df['age'].iloc[i])
        if not str(df['gender'].iloc[i]) in ['male', 'female']:
            wrong_values_gender.append(df['gender'].iloc[i])

    if print_choice == 1:
        print("Different age values : ", set(wrong_values_age), "\n\n")
        print("Different gender values : ", set(wrong_values_gender), "\n\n")

    return wrong_values_age, wrong_values_gender





def read_mp3file(filepath, print_choice=0): # print_choice=1 if you want to print the informations
    audio, sr = librosa.load(filepath)

    if print_choice == 1:
        print("audio : \n", audio, "\n") # audio est un tableau qui contient toutes les intensités du signal
        print("length of audio : ", len(audio))
        print("sr : ", sr) # sr est le taux d'échantillonnage, c'est le nombre d'échantillons par seconde

    return audio, sr





def compute_audio_time_from_filepath(filepath):
    audio, sr = read_mp3file(filepath)

    return compute_audio_time_from_audio(audio, sr)





def compute_audio_time_from_audio(audio, sr):
    audio_time = len(audio) / sr

    return audio_time





def get_validated_audio_times(print_choice=0): # print_choice=1 if you want to print the informations of the function
    # Récupérer le dataframe correspondant au fichier 'validated.tsv'
    load_tsv_files()

    idx = 0

    for i in filenames_list:
        if filenames_list[0] == 'validated.tsv':
            idx = i

    df = filenames_list[idx][1]

    # Parcourir les audios validés
    df_path = df['path']

    if print_choice == 1:
        print("length of df_path : ", len(df_path))

    list_audio_times = []

    for j in tqdm(range(len(df_path)), desc='Processing audios'):
        # Calculer le temps de l'audio pour chaque audio
        filepath = mp3_files_folder + df_path.iloc[j]

        audio_time = round(compute_audio_time_from_filepath(filepath), 1)

        list_audio_times.append(audio_time)

        if print_choice == 1:
            print("audio time in seconds : ", audio_time)

    return list_audio_times





def plot_audio_times_informations():
    list_audio_times = np.array(get_validated_audio_times())

    # Afficher la valeur minimale et maximale
    print("Minimum :", min(list_audio_times))
    print("Maximum :", max(list_audio_times))

    # Tracer la distribution des valeurs
    plt.hist(list_audio_times, bins=500, alpha=0.5, color='blue')

    plt.xlabel('Valeurs')
    plt.ylabel('Fréquence')
    plt.title('Distribution des valeurs')

    plt.show()





def plot_audio_graphs(nbr_audios):
    load_tsv_files()
    wrong_age_values, _ = get_wrong_values(filenames_list[0][0])
    df = filenames_list[0][1]

    df = df[~df['age'].isin(wrong_age_values)]

    random_sample = df.sample(n=nbr_audios)

    for i in range(len(random_sample)):
        file_path = mp3_files_folder + random_sample.iloc[i]['path']

        audio, sr = read_mp3file(file_path)

        duration = librosa.get_duration(y=audio, sr=sr)
        time = np.linspace(0, duration, len(audio))

        plt.figure(figsize=(4, 2))
        plt.plot(time, audio, color='b')
        plt.xlabel('Temps (s)')
        plt.ylabel('Intensité')
        plt.title('Intensité de l\'audio en fonction du temps')
        plt.show()



def main_eda_51():
    load_tsv_files(1)
    discover_tsv_file("validated.tsv")
    #get_wrong_values('validated.tsv', 1)

    #read_mp3file("../cv_corpus_14.0/en/clips_mp3/common_voice_en_37870926.mp3")
    #print("Time in seconds : ", compute_audio_time_from_filepath("../cv_corpus_14.0/en/clips_mp3/common_voice_en_37870926.mp3"))

    #plot_audio_times_informations()

    #plot_audio_graphs(100)

