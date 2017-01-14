

# MODULES REQUIRED________________________________________________________:
from __future__ import print_function
from scipy import ndimage as ndi
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
import numpy as np
import cv2
import os
import csv


# Some variables for later use
num_of_feature_rows_per_vid = 200  # 20 features per image in 10 sec video and according to video 200 audio feature rows
num_of_thetas = 4  # thetas used in kernels for image features
frequencies = (0.10, 0.30, 0.50, 0.70, 0.90)  # frequencies used in kernels for image features
width_of_image = 100  # 100x100 dimension image is used for image features
Winlen = 0.1  # used in audio features. 0.1 means window length is 100ms
Winstep = 0.1  # used in audio features for adjusting audio frame settings

ALL_FEATS = []  # STORE ALL IMAGE AND AUDIO FEATURES

data_write_location = "./data_per_lec/"
data_file_extension = '.txt'
read_location = "./videos_and_audios_used_in_project/"
write_location = "./images_per_sec/"
video_extension = ".mp4"
image_extension = ".png"
audio_extension = ".wav"


# GET THE NAMES OF FILES FROM LOCATION OF YOUR DATA_______________________:
videofiles_name = []
audiofiles_name = []
files_name = []
for unused1, unused2, files in os.walk(read_location):
    for file in files:
        if file.endswith(video_extension):
            videofiles_name.append(file)
        elif file.endswith(audio_extension):
            audiofiles_name.append(file)
        name = file.split(".")
        if name[0] not in files_name:
            files_name.append(name[0])

# Sort all names for similarity
videofiles_name.sort()
audiofiles_name.sort()
files_name.sort()

# IMPORT CLASSIFICATION LABELS AND COMBINE IT WITH DATA:
labels = {}
with open('LABELS.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        labels = row
print (labels)

# EXTRACT AUDIO AND VIDEO FEATURES________________________________________________________________:
# Function for redefining the dimensions of the frame:
def resize_img(file_name, width):
    ratio = int(width) / file_name.shape[1]
    dim = (int(width), int(file_name.shape[0] * ratio))
    resized = cv2.resize(file_name, dim, interpolation=cv2.INTER_AREA)
    return resized

# Function for computing image features:
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

# prepare filter bank kernels with 8 orientations and 5 frequencies for images:
kernels = []
for theta in range(num_of_thetas):
    theta = theta / 4. * np.pi
    for frequency in frequencies:
        kernel = np.real(gabor_kernel(frequency, theta=theta))
        kernels.append(kernel)

# Load haarcascade for face detection:
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img_name = 0  # for naming cropped images
for i, name in enumerate(files_name):
    if videofiles_name[i].startswith(name) and audiofiles_name[i].startswith(name):
        image_feats_per_vid = []
        audio_feats_per_vid = []
        cap = cv2.VideoCapture(read_location+videofiles_name[i])  # Capture video from location
        FPS = int(cap.get(cv2.CAP_PROP_FPS))  # Count the FPS of given video
        print("Working on", name)
        print('Video FPS', FPS)
        per_vid_counter = 0  # for counting 10 images per video
        count = 0  # used with FPS
        while (cap.isOpened()):
            if per_vid_counter == 10:
                break
            ret, frame = cap.read()
            if ret == False:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                face = resize_img(gray[y:y + h, x:x + w], width_of_image)  # Get faces from image
                if face.any() and (count % FPS == 0 or count % FPS == 1):
                    print("Extracting Faces", per_vid_counter)
                    cv2.imwrite(write_location+str(img_name)+image_extension, face)  # write the face image to given directory
                    image = img_as_float(face)
                    image_feats = compute_feats(image, kernels)  # 20 features per image
                    image_feats_per_vid.append(image_feats)
                    per_vid_counter += 1
                    img_name += 1
                    break
            count += 1
        cap.release()
        image_feats_per_vid = np.asarray(image_feats_per_vid)  # Convert the python list into numpy array
        image_feats_per_vid = np.reshape(image_feats_per_vid, (num_of_feature_rows_per_vid,-1))  # Converting 3d array to 2d array
        print ("IMAGE DATA SIZE PER VIDEO =", image_feats_per_vid.shape)


        # Getting audio features
        (rate, signal) = wav.read(read_location + audiofiles_name[i])  # Reading audio file
        mfcc_feat = mfcc(signal, rate, winlen=Winlen, winstep=Winstep)
        mfcc_feat = mfcc_feat[:num_of_feature_rows_per_vid, :]
        fbank_feat = logfbank(signal, rate, winlen=Winlen, winstep=Winstep)
        fbank_feat = fbank_feat[:num_of_feature_rows_per_vid, :]
        audio_feats_per_vid = mfcc_feat
        print ("AUDIO DATA SIZE PER VIDEO =", audio_feats_per_vid.shape)


        # Combine image and audio features with label into one list and attach it to file name using dictionary
        label = []
        for unused in range(num_of_feature_rows_per_vid):
            label.append(int(labels[name]))
        label = np.array(label)
        video_audio = np.column_stack((image_feats_per_vid,audio_feats_per_vid))
        full_feats = np.column_stack((video_audio, label))
        ALL_FEATS.append(full_feats)
        print (ALL_FEATS)
        print("SHAPE OF FINAL DATA (with labels) =", ALL_FEATS[i].shape)


    # SAVE FEATURES TO A VIDEO NAMED FILE
    print("WRITING DATA TO FILE...")
    np.savetxt(data_write_location+name+data_file_extension, ALL_FEATS[i], delimiter=',')
    print("DONE WRITING")
    print("--" * 30)


# THIS LINE SHOULD BE AT BOTTOM ALWAYS
print("**"*10,"END OF PROGRAM","**"*10)