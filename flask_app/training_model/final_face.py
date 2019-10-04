# coding: utf-8

# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

import face_recognition
from sklearn import svm
import os
import pickle
import datetime
import shutil
import pathlib
import h5py


# Training the face recognition with SVC classifier
def train_model():
    print('Start training!!!!')
    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory
    dir_face = "face_training_dataset/"
    train_dir = os.listdir(dir_face)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(dir_face + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(dir_face + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains none or more than faces, print an error message and exit
            if len(face_bounding_boxes) != 1:
                print(
                    person + "/" + person_img + " contains none or more than one faces and can't be used for training.")
                # exit()
            else:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
                print('Finish',person)

    # Create a timestamp for insert in model name
    now = datetime.datetime.now()
    curr_time = str(datetime.datetime.timestamp(now))

    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings, names)
    filename = 'model/' + 'model_' + curr_time + '.sav'

    pickle.dump(clf, open(filename, 'wb'))

    print('Created', filename)
    return filename


# Train new face to model
def train_new_face():
    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory
    dir_face = "C:/Users/Admin/Desktop/final_project/new_face/"
    train_dir = os.listdir(dir_face)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(dir_face + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(dir_face + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains none or more than faces, print an error message and exit
            if len(face_bounding_boxes) != 1:
                print(
                    person + "/" + person_img + " contains none or more than one faces and can't be used for training.")
                exit()
            else:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)

    # Create a timestamp for inserting in model name. Later the timestamp will be used in load_latest_model() function
    curr_time = str(datetime.datetime.now())

    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings, names)

    # Export to h5 model
    filename = 'model_' + curr_time[11:19].replace(':', '_') + '.h5'
    hf = h5py.File(filename, 'w')

    # Export to 
    # pickle.dump(clf, open(filename, 'wb'))
    print('Created', filename)

    # Delete new faces in the training data for new_face
    for person in train_dir:
        shutil.rmtree(dir_face + person)

    return filename


# Load the test image with unknown faces into a numpy array
def test_image_svm(image_name, model):
    test_image = face_recognition.load_image_file(image_name)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)

    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = model.predict([test_image_enc])
        print(*name)


# Go into model directory and find model with latest timestamp
def load_latest_model():
    path = 'C:/Users/Admin/Desktop/final_project/model/'
    model_dir = os.listdir(path)
    latest_model_name = ""
    latest_model_time = 0
    for i, model in enumerate(model_dir):
        curr_time = float(model[6:-4])
        if curr_time > latest_model_time:
            latest_model_time = curr_time
            latest_model_name = model

    trained_model = pickle.load(open(path + latest_model_name, 'rb'))
    print("Latest model is", latest_model_name)
    return trained_model