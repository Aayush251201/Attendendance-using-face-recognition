import cv2
import numpy as np
import os
import pickle
from PIL import Image
from datetime import datetime

def samples():
    name=input("enter your name : ")
    path1='C:/Users/Aayush/PycharmProjects/FaceDetection and Attendance/'+str(name)
    moduleVal = 5
    minBlur = 500
    ##imgW=180
    ##imgH=120

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    num = 0
    count = 0
    ctr = 0

    os.mkdir(path1)

    while True:

        ret, frame = cap.read()
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ctr % moduleVal == 0:
            cv2.imwrite(path1 + '/' + str(num) + ".jpg", imgGray)
            num += 1
        ctr += 1
        count += 1
        cv2.putText(imgGray,str(num), (60,80), cv2.FONT_ITALIC,2 , (0, 255, 0), 3)
        cv2.imshow("img", imgGray)
        if cv2.waitKey(1) & count == 500:
            break



def trainer():
    label_id = {}
    current_id = 0
    x_train = []
    y_label = []
    face_cascade = cv2.CascadeClassifier("images/haarcascade_frontalface_default.xml")
    base_dir = os.path.dirname(os.path.abspath("__file__"))
    image_dir = os.path.join(base_dir, "C:/Users/Aayush/PycharmProjects/FaceDetection and Attendance")
    for root, dir, files in os.walk(image_dir):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path))
                # print(label,path)
                if not label in label_id:
                    label_id[label] = current_id
                    current_id += 1
                id = label_id[label]
                # print(label_id)
                # y_lable.append(lable)
                # x_train.append(path)
                pil_image = Image.open(path).convert("L")
                final_image = pil_image.resize((500, 500), Image.ANTIALIAS)
                image_array = np.array(final_image, "uint8")
                # print(image_array)
                faces = face_cascade.detectMultiScale(image_array, 1.5, 5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_label.append(id)
                    # print(x_train)
                    # print(y_label)

    with open("face_label.pickle", "wb") as f:
        pickle.dump(label_id, f)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(x_train, np.array(y_label))
    recognizer.save("trainer.yml")


def final():
    faceCascade = cv2.CascadeClassifier("images/haarcascade_frontalface_default.xml")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    label = {"person_name": 1}
    with open("face_label.pickle", 'rb') as f:
        og_label = pickle.load(f)
        label = {v: k for k, v in og_label.items()}

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.5, 5)
        for (x, y, w, h) in faces:
            # print(x, y, w, h)
            roi_gray = imgGray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            id, conf = recognizer.predict(roi_gray)
            if conf >= 45:
                print(conf)
                print(label[id])
                obj_name = label[id]
                attendance(obj_name)
                cv2.putText(imgGray, obj_name, (x, y), cv2.FONT_ITALIC, 0.9, (255, 0, 255), 2)

                cv2.rectangle(imgGray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(imgGray, (x, y - 35), (x, y), (255, 0, 0), cv2.FILLED)

            ##cv2.putText(imgGray, "face", (x, y), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
        cv2.imshow("face_detect", imgGray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def attendance(name):

    with open("attendance.csv",'r+') as f:

        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')


print("1) Collect Samples")
print("2) Train model")
print("3) Face detection")
choice = int(input("Enter your choice: "))
if choice == 1:
    samples()
elif choice==2:
    trainer()
elif choice==3:
    final()


