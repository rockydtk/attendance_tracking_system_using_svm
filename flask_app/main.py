from flask import Flask, render_template, request, send_from_directory
import pymysql
import os
import pickle
import face_recognition
import datetime
import training_model # function train model, add new face, predict img path, load latest timestamp model
import db_conn # initiate class database and CRUD command
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)


# Define path to static and upload folder
dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'


# Load trained model
path = 'static\models\model_face.sav'
model = pickle.load(open(path, 'rb'))


@app.route('/')
def landing_page():
    return render_template('login.html')


@app.route('/edit', methods=['POST','GET'])
def edit_page():
    return render_template('edit.html')


@app.route('/back', methods=['POST','GET'])
def back_to_upload_page():
    return render_template('upload.html')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('upload.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(full_name)

        # indices = {0: 'Dung', 1: 'Duong', 2: 'Hai Minh', 3: 'Hoa', 4:'Khiem', 5:'Lan', 6:'Mina', 7:'Minh', 8:'Nguyen', 9:'Phuong', 10:'Sandy', 11:'Tri', 12:'Tu'}

        upload_image = face_recognition.load_image_file(full_name)

        # Find all the faces in the test image using the default HOG-based model
        face_locations = face_recognition.face_locations(upload_image)

        no = len(face_locations)
        print("Number of faces detected: ", no)

        # Predict all the faces in the test image using the trained classifier
        print("Found:")

        # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
        pil_image = Image.fromarray(upload_image)
        # Create a Pillow ImageDraw Draw instance to draw with
        draw = ImageDraw.Draw(pil_image)

        # Create instance to display on predict page
        attend = datetime.date.today()
        class_name = "Fansipan"
        student_names = []
        for i in range(no):
            image_enc = face_recognition.face_encodings(upload_image)[i]

            detected_location = face_locations[i]
            name = model.predict([image_enc])
            student_names.append(name[0])

            # for drawing positions in detected_location
            top, right, bottom, left = detected_location
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name[0])
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            font_path = "C:/Users/Admin/Desktop/final_project/final_web_app/app/static/fonts/font-awesome-4.7.0/fonts/VHARIAL.TTF"
            font = ImageFont.truetype(font_path, 20)
            draw.text((left + 20, bottom - text_height - 10), str(name[0]), fill=(255, 255, 255, 255), font=font, align = 'center')

                       
        img_bounding_box = os.path.join(UPLOAD_FOLDER, "image_with_boxes.jpg")
        pil_image.save(img_bounding_box)

        # predicted_names = db.list_attended_students()
        predicted_names = {'student_names': student_names, 'class_name':class_name, 'attend': attend}

        return render_template('predict.html', 
            result=predicted_names, image_file_name = "image_with_boxes.jpg",content_type='application/json')


@app.route('/student', methods=['POST','GET'])
def get_student():
    conn = db_conn.Database()
    predicted_names = conn.list_attended_students()
    return render_template('student.html', 
        result=predicted_names,content_type='application/json')


if __name__ == '__main__':
    app.run()