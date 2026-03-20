# GENERATION PARAMETERS
#RAND_NOISE_DIR_COEFF = 0.6 # Probability of a type of sound to be selected as noise
#SOUND_DURATION = 5000 # Milliseconds

#DATASET_SIZE = 20000

#VELOCITY_RANGE = range(-100, 101, 1)
#INTENSITY_RANGE = range(-15, 16, 5)

NOISES_PER_SUBFOLDER = {
    "brake": 1,
    "dog": 3,
    "drill": 3,
    "engine": 6,
    "horn": 2,
    "music": 1,
    "nature": 2,
    "rain": 1,
    "voice": 2
}

AMBULANCE_PATH = "sounds/ambulances/"
PURE_SIREN_PATH = "sounds/ambulances/pure_ambulance.wav"
NOISE_PATH = "sounds/noises/"
DATASET_SOUND_PATH = "dataset/sounds/"
DATASET_IMAGES_PATH = "dataset/images/"