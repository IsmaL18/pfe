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
mp3_files_folder = "../cv-corpus_5.1/fr/clips/"

hop_length = 512  # the default spacing between frames
n_fft = 255  # number of samples




def load_tsv_files(print_choice=0): # print_choice=1 if you want to print the informations
    print("\n\n\n\n\n---------- LOAD TSV FILES ----------")
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





def get_mel_spectrogram(audio, sr, print_choice=0, plot_choice=0, num=0):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

    if plot_choice == 1:
        fig, ax = plt.subplots(figsize=(20, 7))
        librosa.display.specshow(spectrogram, sr=sr, cmap='cool', hop_length=hop_length)
        ax.set_xlabel('Time', fontsize=15)
        if num == 0:
            ax.set_title('Spectrogram', size=20)
        else:
            ax.set_title('Spectrogram File N°' + str(num), size=20)
        plt.colorbar()
        plt.show()

    if print_choice == 1:
        print("\n---------- Mel Spectrogram ----------")
        print("Mel Spectrogram type : ", type(spectrogram))
        print("Mel Spectrogram shape : ", spectrogram.shape)

    return spectrogram





def get_db_spectrogram(audio, sr, print_choice=0, plot_choice=0, num=0):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    if plot_choice == 1:
        fig, ax = plt.subplots(figsize=(20, 7))
        librosa.display.specshow(db_spectrogram, sr=sr, cmap='cool', hop_length=hop_length)
        ax.set_xlabel('Time', fontsize=15)
        if num == 0:
            ax.set_title('db Spectrogram', size=20)
        else:
            ax.set_title('dB Spectrogram File N°' + str(num), size=20)
        plt.colorbar()
        plt.show()

    if print_choice == 1:
        print("\n---------- dB Spectrogram ----------")
        print("dB Spectrogram type : ", type(db_spectrogram))
        print("dB Spectrogram shape : ", db_spectrogram.shape)

    return db_spectrogram





def get_mccs(audio, sr, print_choice=0, plot_choice=0, num=0):
    mfccs = librosa.feature.mfcc(y=audio, n_fft=n_fft,hop_length=hop_length,n_mfcc=128)

    if plot_choice == 1:
        fig, ax = plt.subplots(figsize=(20, 7))
        librosa.display.specshow(mfccs, sr=sr, cmap='cool', hop_length=hop_length)
        ax.set_xlabel('Time', fontsize=15)
        if num == 0:
            ax.set_title('MFCC', size=20)
        else:
            ax.set_title('MFCC File N°'+str(num), size=20)
        plt.colorbar()
        plt.show()

    if print_choice == 1:
        print("\n---------- MFCCs ----------")
        print("MFCCs type : ", type(mfccs))
        print("MFCCs shape : ", mfccs.shape)

    return mfccs





def plot_graphs_one_audio(filepath):
    audio, sr = read_mp3file(mp3_files_folder + filepath)

    get_mccs(audio, sr, plot_choice=1, print_choice=1)
    get_mel_spectrogram(audio, sr, plot_choice=1, print_choice=1)
    get_db_spectrogram(audio, sr, plot_choice=1, print_choice=1)





def plot_graphs_audios(n_audios):
    load_tsv_files()

    print("\n\n\n\n\n---------- PLOT GRAPHS AUDIOS ----------")

    df = filenames_list[0][1].sample(n=n_audios)

    for idx in range(len(df)):
        audio, sr = read_mp3file(mp3_files_folder + df.iloc[idx]['path'])

        print("\n\n---------- FILE N°",  idx+1, " ----------")
        print("Age = ", df.iloc[idx]['age'])
        print("Gender = ",  df.iloc[idx]['gender'])
        print("Audio time : ", compute_audio_time_from_audio(audio, sr))
        get_mccs(audio, sr, plot_choice=1, print_choice=1, num=idx+1)
        get_mel_spectrogram(audio, sr, plot_choice=1, print_choice=1,  num=idx+1)
        get_db_spectrogram(audio, sr, plot_choice=1, print_choice=1, num=idx+1)





def main_eda_spec():
    #plot_graphs_one_audio("common_voice_fr_17299400.mp3")
    plot_graphs_audios(3)

