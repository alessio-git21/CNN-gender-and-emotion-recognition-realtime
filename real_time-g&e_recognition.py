from tkinter import Frame
from cv2 import threshold
import tensorflow as tf
from tensorflow import keras
#import pandas as pd
import cv2
import numpy as np

cap=cv2.VideoCapture(0)
codec=cv2.VideoWriter_fourcc(*'MJPG')

#load cascade classifier
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #face detection

#load CNN
gender_model=keras.models.load_model("gender_vgg16model.h5")
emotion_model=keras.models.load_model("my_emotion_model.h5")

bw=False
em=False

while(cap.isOpened()):

    _,frame=cap.read()

    if(bw): # if cascade classifier and gender recognition are activated

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #from BGR to grayscale (the haar cascade classifier wants grayscale images)

        faces = face_cascade.detectMultiScale(gray,1.1,8) # cascade classifier



        for (x,y,w,h) in faces: #for each bounding box whithin there is a face
            


            rectangle = frame[y-25:y+h+25,x-25:x+h+25] #take the face (RGB)

            try:

                #preprocesing for gender recognition
                
                input_CNN=cv2.resize(rectangle,(224,224))
                input_CNN=cv2.cvtColor(input_CNN, cv2.COLOR_BGR2RGB)
                input_CNN=keras.applications.vgg16.preprocess_input(input_CNN)
                input_CNN=input_CNN.reshape(1,224,224,3)

                #gender prediction

                gender=gender_model.predict(input_CNN)
                gender=gender[0][0]

                threshold=0.5

                gender_percentage=round(gender*100,1) if gender>threshold else round((1-gender)*100,1)

                gender_label="sex: woman" if gender>threshold else "sex: man"

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 

                if(em==False): #if emotion recognition is NOT activated, write only gender results
                    cv2.rectangle(frame,(x,y-30),(x+300,y),(0,255,0),cv2.FILLED)
                    cv2.putText(frame,gender_label+f" ({gender_percentage}%)",(x,y-5),cv2.FONT_HERSHEY_PLAIN,1.4,(0,0,0),1)

                elif(em): #if also emotion recognition is activated


                    emotion_dict={0:'angry',1:'happy',2:'neutral',3:'sad',4:'surprised'}
                    emotions=['angry','happy','neutral','sad','surprised']

                    #preprocessing for emotion recognition

                    em_rectangle=gray[y:y+h,x:x+h] #emotion recognition wants grayscale images
                    input_emCNN=cv2.resize(em_rectangle,(48,48))
                    input_emCNN=input_emCNN/255.
                    input_emCNN=input_emCNN.reshape(1,48,48,1)



                    #emotion prediction

                    emotion=emotion_model.predict(input_emCNN)


                    xd = dict(   zip( emotions,list(emotion[0]) )   )
                    sorted_x= dict(sorted(xd.items(), reverse=True, key=lambda item: item[1]))

                    emotion_label='emotion: '+ emotion_dict[np.argmax(emotion)]
                    emotion_percentage=round(emotion.max()*100,1) 

                    cv2.rectangle(frame,(x,y-60),(x+330,y),(0,255,0),cv2.FILLED)
                    cv2.putText(frame,gender_label+f" ({gender_percentage}%)",(x,y-35),cv2.FONT_HERSHEY_PLAIN,1.4,(0,0,0),1) #write gender results
                    cv2.putText(frame,emotion_label+f" ({emotion_percentage}%)",(x,y-5),cv2.FONT_HERSHEY_PLAIN,1.4,(0,0,0),1) #write emotion results



            except cv2.error:
                print('cv2 reshape error')
                continue
            
        cv2.imshow('gender recognition',frame)
        
    elif(bw==False):
        cv2.imshow('gender recognition',frame)



    k=cv2.waitKey(1)

    if(k==ord('g')):
        bw= not bw
        if(bw):
            print('classification: on')
        elif(bw!=True):
            print('classification: off')

    if(k==ord('e')):
        em= not em
        if(em):
            print('emotion recognition: on')
        elif(em!=True):
            print('emotion recognition: off')

    elif(k==ord("q")):
        break


cap.release()
cv2.destroyAllWindows()

 