import flask
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import sys
import filetype
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

FACE_PROTO = "weights/deploy.prototxt.txt"

FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# The gender model architecture
GENDER_MODEL = 'weights/deploy_gender.prototxt'

# The gender model pre-trained weights
GENDER_PROTO = 'weights/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

AGE_MODEL = 'weights/deploy_age.prototxt'

AGE_PROTO = 'weights/age_net.caffemodel'

AGE_INTERVALS = ['(00, 02)', '(04, 06)', '(08, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

frame_width = 1280
frame_height = 720

# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

# Load gender prediction model# Home page - displays the form for uploading images
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)
f_int1 = 0
s_int1 = 0
f_int2 = 0
s_int2 = 0



@app.route('/')
def index():
    return render_template('app.html')


# Result page - displays the uploaded images
@app.route('/', methods=['POST'])
def results():
    # Get the uploaded files from the request
    model = tf.keras.models.load_model('new_model/')
    file1 = request.files['image1']
    file2 = request.files['image2']

    # Use the model to classify a new input image
    # Save the uploaded files to the uploads folder
    file1.save(os.path.join(app.config['UPLOAD_FOLDER'], file1.filename))
    file2.save(os.path.join(app.config['UPLOAD_FOLDER'], file2.filename))

    # Get the URLs of the uploaded files
    file1_url = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    file2_url = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)

    def get_faces(frame, confidence_threshold=0.5):
        # convert the frame into a blob to be ready for NN input
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
        # set the image as input to the NN
        face_net.setInput(blob)
        # perform inference and get predictions
        output = np.squeeze(face_net.forward())
        # initialize the result list
        faces = []
        # Loop over the faces detected
        for i in range(output.shape[0]):
            confidence = output[i, 2]
            if confidence > confidence_threshold:
                box = output[i, 3:7] * \
                      np.array([frame.shape[1], frame.shape[0],
                                frame.shape[1], frame.shape[0]])
                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(int)
                # widen the box a little
                start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = 0 if end_x < 0 else end_x
                end_y = 0 if end_y < 0 else end_y
                # append to our list
                faces.append((start_x, start_y, end_x, end_y))
        return faces

    def display_img(title, img):

        # Display Image on screen
        cv2.imshow(title, img)
        # Mantain output until user presses a key
        cv2.waitKey(0)
        # Destroy windows when user presses a key
        cv2.destroyAllWindows()

    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        # resize the image
        return cv2.resize(image, dim, interpolation=inter)

    def get_gender_predictions(face_img):
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
        )
        gender_net.setInput(blob)
        return gender_net.forward()

    def get_age_predictions(face_img):
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=MODEL_MEAN_VALUES, swapRB=False
        )
        age_net.setInput(blob)
        return age_net.forward()

    def predict_age_and_gender(input_path: str):
        """Predict the gender of the faces showing in the image"""

        # Read Input Image
        img = cv2.imread(input_path)

        # Take a copy of the initial image and resize it
        frame = img.copy()
        height = frame.shape[0]
        width = frame.shape[1]

        # new_height = 400
        # ratio = new_height / height # (or new_height / height)
        # new_width = int(width * ratio)

        # dimensions = (new_width, new_height)
        # frame = cv2.resize(frame, dimensions, interpolation=cv2.INTER_LINEAR)

        frame = image_resize(frame, height=400)

        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        # predict the faces
        faces = get_faces(frame)
        n_faces = len(faces)

        if n_faces == 0:
            # print("Not Valid Passport Image")
            return 0

        elif n_faces == 1:
            # Loop over the faces detected
            # for idx, face in enumerate(faces):
            for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
                face_img = frame[start_y: end_y, start_x: end_x]
                age_preds = get_age_predictions(face_img)
                gender_preds = get_gender_predictions(face_img)
                i = gender_preds[0].argmax()
                gender = GENDER_LIST[i]
                gender_confidence_score = gender_preds[0][i]
                i = age_preds[0].argmax()
                age = AGE_INTERVALS[i]
                age_confidence_score = age_preds[0][i]
                # Draw the box
                label = f"{gender}-{gender_confidence_score * 100:.1f}%, {age}-{age_confidence_score * 100:.1f}%"
                # label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
                print(label)

            #     yPos = start_y - 15
            #     while yPos < 15:
            #         yPos += 15
            #     box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
            #     cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            #     # Label processed image
            #     font_scale = 0.54
            #     cv2.putText(frame, label, (start_x, yPos),
            #                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
            
            # Display processed image
            # display_img('face', frame)
            # print("Valid Passport Image")
            
            # uncomment if you want to save the image
            # cv2.imwrite("face_detected.jpg", frame)
            # Cleanup
            # cv2.destroyAllWindows()

            return 1
        else:
            # print("Not Valid Passport Image")
            return 0

    def detect_signature(input_path: str):

        #New model with feature prediction
        model = load_model('new_model/')
        # base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        base_model = load_model('base_model/')
        # Define a function to extract features from an input image
        def extract_features(image_path):
            # Load the image and resize it to match the input shape of the pre-trained model
            image = load_img(image_path, target_size=(224, 224))
            
            # Convert the image to a NumPy array and preprocess it for the pre-trained model
            image_array = img_to_array(image)
            image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
            
            # Use the pre-trained model to extract features from the image
            features = base_model.predict(np.array([image_array]))
            
            # Flatten the feature tensor into a vector
            flattened_features = np.ravel(features)
            
            # Return the flattened features
            return flattened_features
        image = input_path
        new_image_features = extract_features(image)
            
        # Use the binary classification model to predict whether the new image is a signature or non-signature image
        predicted_prob = model.predict(np.array([new_image_features]))[0]
        print(predicted_prob) 
        # Print the predicted class and probability
        if predicted_prob > 0.5:
            print("The new image is classified as a signature image.")
            return 1
        else:
            print("The new image is classified as a non-signature image.")
            return 0
        # print(f"The probability of the new image being a signature image is {predicted_prob:.2f}.")

    input_path1 = file1_url
    input_path2 = file2_url

    f_int1 = predict_age_and_gender(input_path1)
    s_int1 = detect_signature(input_path1)

    f_int2 = predict_age_and_gender(input_path2)
    s_int2 = detect_signature(input_path2)

    f_arr = [f_int1, s_int1, f_int2, s_int2]

    print(f_arr)

    if f_arr == [1, 0, 0, 1]:
        print("First Image is Face and Second is Signature")
        return render_template('app.html', file1_url=file1_url, file2_url=file2_url, label1="Face Image:", label2="Sign Image:")

    elif f_arr == [0, 1, 1, 0]:
        print(("First Image is Signature and Second is Face"))
        return render_template('app.html', file1_url=file2_url, file2_url=file1_url, label1="Face Image:", label2="Sign Image:")
    
    elif f_arr == [1, 0, 1, 0]:
            print(("Both are Face Images"))
            return render_template('app.html', file1_url=file1_url,file2_url=file2_url,error_msg = "Reupload with a sign image and correct face image", label1="Face Image 1:", label2="Face Image 2:")
    
    elif f_arr == [0, 1, 0, 1]:
            print(("Both are Signature Images"))
            return render_template('app.html', file1_url=file1_url,file2_url=file2_url,error_msg = "Reupload with a face image and correct sign image", label1="Sign Image 1:", label2="Sign Image 2:")

    elif f_arr == [1, 0, 0, 0]:
            print(("There is no signature image, only face image"))
            return render_template('app.html', file1_url=file1_url,file2_url=file2_url,error_msg = "Reupload with sign image", label1="Face Image:", label2="Wrong Image:")

    elif f_arr == [0, 0, 1, 0]:
            print(("There is no signature image, only face image"))
            return render_template('app.html', file1_url=file2_url,file2_url=file1_url,error_msg = "Reupload with sign image", label1="Face Image:", label2="Wrong Image:")

    elif f_arr == [0, 1, 0, 0]:
            print(("There is no face image, only signature image"))
            return render_template('app.html', file1_url=file2_url,file2_url=file1_url,error_msg = "Reupload with face image", label1="Wrong Image:", label2="Sign Image:")

    elif f_arr == [0, 0, 0, 1]:
            print(("There is no face image, only signature image"))
            return render_template('app.html', file1_url=file1_url,file2_url=file2_url,error_msg = "Reupload with face image", label1="Wrong Image:", label2="Sign Image:")

    else:
        print("Invalid Images")
        return render_template('app.html', file1_url=file2_url, file2_url=file1_url, error_msg = "Both are invalid images", label1="Image 1:  ", label2="Image 2:")

if __name__ == '__main__':
    app.run(debug=True)