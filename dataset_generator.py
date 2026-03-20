import os
import time
import random
import librosa
import json
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.fft import fft, fftfreq
from tqdm import tqdm

from dataset_parameters import *

def calculateFrequency(sound: AudioSegment) -> float:
    # Convert audio to numpy array of samples
    samples = np.array(sound.get_array_of_samples())

    # Normalize the samples to range [-1, 1]
    if sound.channels == 2:  # If stereo, we take only one channel (left or right)
        samples = samples[::2]

    # Perform FFT on the audio samples
    N = len(samples)  # Number of samples
    sample_rate = sound.frame_rate  # Frame rate (samples per second)

    # Perform FFT
    freqs = fftfreq(N, d=1/sample_rate)
    spectrum = np.abs(fft(samples))

    # Get the positive frequencies (we discard the negative ones)
    pos_freqs = freqs[:N//2]
    pos_spectrum = spectrum[:N//2]

    # Find the frequency with the highest amplitude (dominant frequency)
    dominant_freq_index = np.argmax(pos_spectrum)
    dominant_freq = pos_freqs[dominant_freq_index]

    return dominant_freq

def generateDoppler(sound : AudioSegment, velocities: list) -> list:
    frequency = calculateFrequency(sound)
    v_sound = 343.4
    output = []

    for v in velocities:
        dp_frequency = frequency * (v_sound / (v_sound + v))
        frame_rate = int(sound.frame_rate * (dp_frequency / frequency))

        gen_sound = sound._spawn(sound.raw_data, overrides={"frame_rate": frame_rate})

        output.append((gen_sound, v))

        #filename = f"xdd/doppler_v{v}_siren{str(siren)}.wav"
        #gen_sound.export(filename, format="wav")

    return output

def changeVolume(sound : AudioSegment, intensity : float) -> AudioSegment:
    return sound + intensity

def mixSounds(track1 : AudioSegment, track2 : AudioSegment, i1: float, i2: float) -> tuple:
    track1 = changeVolume(track1, i1)
    track2 = changeVolume(track2, i2)
    mixed = track1.overlay(track2)
    return mixed

def matchDuration(reference : AudioSegment, target : AudioSegment) -> AudioSegment:
    rd = len(reference)
    td = len(target)

    if td > rd:
        target = target[:rd]
    elif td < rd:
        repeat_count = rd // td
        reminder = rd % td
        target = (target * repeat_count) + target[:reminder]

    return target

def cutDuration(mixed : AudioSegment, duration : int):
    track_duration = len(mixed)

    if track_duration < duration:
        padding = AudioSegment.silent(duration = duration - track_duration)
        mixed = mixed + padding
    else:
        rand_start = random.randint(0, track_duration - duration)
        mixed = mixed[rand_start:rand_start + duration]

    return mixed

def importNoises(noise_path : str) -> dict:
    noises = {}
    directories = [dir for dir in os.listdir(noise_path) if os.path.isdir(os.path.join(noise_path, dir))]

    for dir in directories:
        noises[dir] = []
        for file in os.listdir(os.path.join(noise_path, dir)): #  if file.endswith((".wav", ".mp3"))
            try:
                noise = AudioSegment.from_file(os.path.join(noise_path, dir, file))
                noises[dir].append(noise)
            except Exception as e:
                print(f"Error extracting sound from {dir}: {e}")

    return noises

def noisesFromSubfolders(noises : dict, rand_noise_dir_coeff) -> list:
    sel_noises = []
        
    for cat in noises:
        try:
            rnd = random.random()
            if rnd <= rand_noise_dir_coeff:
                files = noises[cat]
                num_noises = NOISES_PER_SUBFOLDER.get(os.path.basename(cat), 0) # Establish how many noises to get from each folder
                rand_sound = random.randint(1, num_noises)

                for _ in range(rand_sound):
                    noise = random.choice(files)
                    sel_noises.append(noise)

        except Exception as e:
            print(f"Error extracting sound from {dir}: {e}")

    return sel_noises

def augmentSound(sound : AudioSegment, noises : dict, intensities : list, duration : int, rand_noise_dir_coeff) -> AudioSegment:
    mixed = changeVolume(sound, random.choice(intensities))
    sel_noises = noisesFromSubfolders(noises, rand_noise_dir_coeff)
    
    for noise in sel_noises:
        try:
            noise = matchDuration(mixed, noise)

            mixed = mixSounds(mixed, noise, 0, random.choice(intensities))

            final = cutDuration(mixed, duration)

        except Exception as e:
            print(f"Error mixing noise: {e}")

    return final

def saveWavFormat(sound : AudioSegment, id : int, path : str) -> None:
    os.makedirs(path, exist_ok=True)
    sound.export(path + f"/sound_{id}.wav", format = "wav")

def savePngFormat(spectrogram : np.ndarray, id : int, path : str) -> None:
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram, cmap='gray')
    plt.savefig(path + "/mix_"+str(id)+".jpg", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def generateMelSpectrogram(sound : AudioSegment) -> np.ndarray:
    y = np.array(sound.get_array_of_samples(), dtype=np.float32)
    sr = sound.frame_rate
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
    return mel_spec_db

def saveData(spectrogram : np.ndarray, file_name : str, path : str):
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram, cmap='gray')
    plt.savefig(path + "/" + file_name, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def datasetParameters():
    dataset_size = int(input("Enter size of dataset: "))
    rand_noise_dir_coeff = float(input("Enter probability on selecting noise -> <(0.0, 1.0)>: "))
    sound_duration = int(input("Enter sound duration (milliseconds) -> <time>: "))
    
    velocities = tuple(map(int, input("Enter velocity range (default [-100 101 1]) -> <start> <stop> <step>: ").split()))
    startV, stopV, stepV = velocities if len(velocities) == 3 else (-100, 101, 1)
    velocity_range = range(startV, stopV, stepV)

    intensities = tuple(map(int, input("Enter intensity range (default [-15 16 5]) -> <start> <stop> <step>: ").split()))
    startI, stopI, stepI = intensities if len(intensities) == 3 else (-15, 16, 5)
    intensity_range = range(startI, stopI, stepI)

    return dataset_size, rand_noise_dir_coeff, sound_duration, velocity_range, intensity_range

def generateDataset(pure_siren_path : str, noise_path : str, data_sound_path : str, data_image_path : str) -> tuple:
    dataset_size, rand_noise_dir_coeff, sound_duration, velocity_range, intensity_range = datasetParameters()
    start = time.time()

    siren = AudioSegment.from_wav(pure_siren_path)
    empty_noise = AudioSegment.silent(sound_duration * 2)
    siren_dopplers = generateDoppler(siren, velocity_range)
    noises = importNoises(noise_path)

    datas, classes, speeds, file_names = [], [], [], []
    labels = {}

    loop = tqdm(range(dataset_size), total = dataset_size, leave = True)

    for i in loop:
        try:
            file_name = (f"data_{i}.jpg")

            match i:
                case _ if i < (dataset_size // 2):
                    siren, speed = random.choice(siren_dopplers)

                    # Augment siren sound with random noises
                    mixed = augmentSound(siren, noises, intensity_range, sound_duration, rand_noise_dir_coeff)

                    # Generate Mel spectrogram
                    data = generateMelSpectrogram(mixed)
                    
                    classes.append(1)
                    speeds.append(speed)

                case _ if i >= (dataset_size // 2):
                    # Augment noise sound
                    mixed = augmentSound(empty_noise, noises, intensity_range, sound_duration, rand_noise_dir_coeff)

                    # Generate Mel spectrogram
                    data = generateMelSpectrogram(mixed)

                    classes.append(0)
                    speeds.append(0)

            file_names.append(file_name)
            datas.append(data)

            saveData(data, file_name, data_image_path)

        except Exception as e:
            print(f"Error creating dataset track: {e}")

    for img, cls, spd in zip(file_names, classes, speeds):
        labels[img] = {"class": cls, "speed": spd}

    with open("dataset/labels/labels.json", "w") as lab_file:
        json.dump(labels, lab_file, indent=4)

    end = time.time()
    print(f"Dataset generation completed in {end - start} seconds!")

    return (datas, classes, speeds)