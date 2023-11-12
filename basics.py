import cv2
import numpy as np
import face_recognition

imgEtte=face_recognition.load_image_file('Test Images/ette-1.jpg')

width = int(imgEtte.shape[1] * 20 / 100)
height = int(imgEtte.shape[0] * 20 / 100)

imgEtte=cv2.resize(imgEtte, (width,height), interpolation=cv2.INTER_AREA)
imgEtte=cv2.cvtColor(imgEtte, cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgEtte)[0]
encodeloc=face_recognition.face_encodings(imgEtte)[0]
cv2.rectangle(imgEtte,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)



imgTest=face_recognition.load_image_file('Test Images/apurba.jpg')

width = int(imgTest.shape[1] * 20 / 100)
height = int(imgTest.shape[0] * 20 / 100)

imgTest=cv2.resize(imgTest, (width,height), interpolation=cv2.INTER_AREA)
imgTest=cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

facelocTest=face_recognition.face_locations(imgTest)[0]
encodelocTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)


result=face_recognition.compare_faces([encodeloc],encodelocTest)
faceDis=face_recognition.face_distance([encodeloc],encodelocTest)
#print(result,faceDis)

cv2.putText(imgTest, f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1, (255,0,255),2)
cv2.imshow('Ette',imgEtte)
cv2.imshow('Ette Test',imgTest)
cv2.waitKey(0)