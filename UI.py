from tkinter import *
from tkinter import ttk
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join


class res:
    def add_samples(name):
        os.mkdir('C:/Users/Aayush/PycharmProjects/FaceDetection and Attendance/Samples/' + str(name))

        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def face_extractor(img):

            faces = face_classifier.detectMultiScale(img, 1.3, 5)

            if faces is ():
                return None

            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]

            return cropped_face

        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (200, 200))

                file_name_path = 'C:/Users/Aayush/PycharmProjects/FaceDetection and Attendance/Samples/' + str(name) + '/' + str(count) + '.jpg'
                cv2.imwrite(file_name_path, face)

                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face)
            else:
                print("Face not Found")
                pass

            if cv2.waitKey(1) == 13 or count == 100:
                break

        cap.release()
        cv2.destroyAllWindows()


root = Tk()

root.geometry("1350x700+0+0")
root.title("Facial Recognition Attendance System")

title = Label(root, text="Facial Recognition Attendance System", bd=10, relief=GROOVE ,font = ("times new roman", 40, "bold"), bg='#3498DB', fg='black')
title.pack(side=TOP, fill=X)

name_var = StringVar()
id_var = StringVar()
mail_var = StringVar()
gender_var = StringVar()
contact_var = StringVar()
dob_var = StringVar()
address_var = StringVar()

frame1 =Frame(root, bd=4, relief=RIDGE, bg='#3498DB')
frame1.place(x=20, y=100, width=500, height=650)

frame2 =Frame(root, bd=4, relief=RIDGE, bg='#3498DB')
frame2.place(x=550, y=100, width=800, height=650)

frame1_title = Label(frame1, text="Fill the student details" ,font = ("times new roman", 30, "bold"), bg='#3498DB', fg='white')
frame1_title.grid(row=0, columnspan=2, pady=20)

name = Label(frame1, text="Name:" ,  font = ("times new roman", 20, "bold"), bg='#3498DB', fg='white')
name.grid(row=1, column=0, pady=10, padx=20, sticky='w')

txt_name = Entry(frame1, textvariable = name_var, font = ("times new roman", 15, "bold"), bd=5, relief=GROOVE)
txt_name.grid(row=1, column=1, pady=10, padx=20, sticky='w')

id = Label(frame1, text="College ID:" ,font = ("times new roman", 20, "bold"), bg='#3498DB', fg='white')
id.grid(row=2, column=0, pady=10, padx=20, sticky='w')

txt_id = Entry(frame1, textvariable = id_var,  font = ("times new roman", 15, "bold"), bd=5, relief=GROOVE)
txt_id.grid(row=2, column=1, pady=10, padx=20, sticky='w')

email = Label(frame1, text="E-mail:" ,font = ("times new roman", 20, "bold"), bg='#3498DB', fg='white')
email.grid(row=3, column=0, pady=10, padx=20, sticky='w')

txt_email = Entry(frame1, textvariable = mail_var, font = ("times new roman", 15, "bold"), bd=5, relief=GROOVE)
txt_email.grid(row=3, column=1, pady=10, padx=20, sticky='w')

gender = Label(frame1, text="Gender:" ,font = ("times new roman", 20, "bold"), bg='#3498DB', fg='white')
gender.grid(row=4, column=0, pady=10, padx=20, sticky='w')

combo_gender = ttk.Combobox(frame1, textvariable = gender_var, font = ("times new roman", 13, "bold"), state='readonly')
combo_gender['values'] = ('Male', 'Female', 'Other')
combo_gender.grid(row=4, column=1, pady=10, padx=20)

contact = Label(frame1, text="Contact No.:" ,font = ("times new roman", 20, "bold"), bg='#3498DB', fg='white')
contact.grid(row=5, column=0, pady=10, padx=20, sticky='w')

txt_contact = Entry(frame1, textvariable = contact_var, font = ("times new roman", 15, "bold"), bd=5, relief=GROOVE)
txt_contact.grid(row=5, column=1, pady=10, padx=20, sticky='w')

dob = Label(frame1, text="D.O.B:" ,font = ("times new roman", 20, "bold"), bg='#3498DB', fg='white')
dob.grid(row=6, column=0, pady=10, padx=20, sticky='w')

txt_dob = Entry(frame1, textvariable = dob_var, font = ("times new roman", 15, "bold"), bd=5, relief=GROOVE)
txt_dob.grid(row=6, column=1, pady=10, padx=20, sticky='w')

address = Label(frame1, text="Address:" ,font = ("times new roman", 20, "bold"), bg='#3498DB', fg='white')
address.grid(row=7, column=0, pady=10, padx=20, sticky='w')

txt_address = Text(frame1, width=30, height=4, font=("t", 10))
txt_address.grid(row=7, column=1, pady=10, padx=20, sticky='w')

btn_frame = Frame(frame1, bd = 3, relief=RIDGE, bg='white')
btn_frame.place(x=35, y=530, width=400, height=70)


Getbtn = Button(btn_frame, text="get", command=lambda: res.add_samples(name_var.get()+id_var.get()))
Getbtn.grid(row=0, columnspan=2)


root.mainloop()