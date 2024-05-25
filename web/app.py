from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
from keras import models
from PIL import Image as Image1

app = Flask(__name__)

# Указываем папку для сохранения загруженных изображений
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def predict(filename):
    # Названия файлов модели для определения лица человека
    FACE_PROTO = "../deploy.prototxt.txt"
    FACE_MODEL = "../res10_300x300_ssd_iter_140000_fp16.caffemodel"

    # Создадим карты возрастов и гендеров
    GENDER_LIST = ['Female', 'Male']
    AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                     '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

    frame_width = 1280
    frame_height = 720

    # Загружаем модели
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    gender_net = models.load_model('../gm')
    age_net = models.load_model('../am')

    # функция для поиска лиц на фото
    def get_faces(frame):
        # преобразовать изображение в двоичный объект, чтобы он был готов к вводу NN
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
        # установить изображение в качестве входных данных для NN
        face_net.setInput(blob)
        # получаем предсказание
        output = np.squeeze(face_net.forward())
        # инициализировать список результатов
        faces = []
        # Цикл по обнаруженным лицам
        for i in range(output.shape[0]):
            confidence = output[i, 2]
            if confidence > 0.5:
                box = output[i, 3:7] * \
                      np.array([frame.shape[1], frame.shape[0],
                                frame.shape[1], frame.shape[0]])
                # преобразовать в целые числа
                start_x, start_y, end_x, end_y = box.astype(np.int)
                # немного расширить рамку
                start_x, start_y, end_x, end_y = start_x - \
                                                 10, start_y - 10, end_x + 10, end_y + 10
                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = 0 if end_x < 0 else end_x
                end_y = 0 if end_y < 0 else end_y
                # добавляем в массив
                faces.append((start_x, start_y, end_x, end_y))
        return faces

    # функция для изменения размера
    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=inter)

    def get_gender_predictions(face_img):
        return gender_net.predict(face_img, batch_size=32)

    def get_age_predictions(face_img):
        return age_net.predict(face_img, batch_size=32)


    # path = ['q.jpg', 'i.jfif', 'pu.jpg', 'elena.jpg', 'r1.jpg', 'r2.jpg', 'p.jpg']

    i = filename
    # Загружаем изображение для предсказания
    img = cv2.imread(i)

    # Преобразуем изображение в массив
    train_images = []
    image1 = Image1.open(i)
    image1 = image1.resize((300, 300))  # Изменим размер изображения
    data = np.asarray(image1)
    train_images.append(data)
    train_images = np.asarray(train_images)

    frame = img.copy()
    if frame.shape[1] > frame_width:
        frame = image_resize(frame, width=frame_width)
    # поиск лиц на фото
    faces = get_faces(frame)
    # Цикл по обнаруженным лицам
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y: end_y, start_x: end_x]

        # предсказать возраст и пол
        age_preds = get_age_predictions(train_images)
        gender_preds = get_gender_predictions(train_images)

        # обрабатываем предсказание
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]
        i = age_preds[0].argmax()
        age = AGE_INTERVALS[i]
        age_confidence_score = age_preds[0][i]

        # добавляем рамку
        label = f"{gender}-{gender_confidence_score * 100:.1f}%, {age}-{age_confidence_score * 100:.1f}%"
        print(label)
        yPos = start_y - 15
        while yPos < 15:
            yPos += 15
        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        font_scale = 0.4
        cv2.putText(frame, label, (start_x, yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)

        # выводим изображение
        window_name = 'image'
        cv2.imshow(window_name, frame)
        # сохраняем изображение
        cv2.imwrite(filename, frame)
        #return frame
        break

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    input = request.files['file']
    uploads_dir = "upload"
    filename = os.path.join(os.getcwd(), uploads_dir, secure_filename(input.filename))
    input.save(filename)

    if input.filename == '':
        return redirect(request.url)

    print("Saving file to:", filename)  # Добавим вывод отладочной информации
    predict(filename)
    return redirect(url_for('show_image', filename=input.filename))


if __name__ == '__main__':
    app.run(debug=True)
