import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='Test Images'
images=[]
classNames=[]

imgLst=os.listdir(path)
for i in imgLst:
    curImg=cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])

def findEncoding(imgs):
    encode_lst=[]
    for img in imgs:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encode_lst.append(encode)
    return encode_lst

def markAttendence(name):
    with open('attendence.csv', 'r+') as f:
        lst=f.readlines()
        namelst=[]
        for line in lst:
            entry=line.split(',')
            namelst.append(entry[0])

        if name not in namelst:
            now=datetime.now()
            dtstring=now.strftime('%H: %M: %S')
            f.writelines(f'\n{name},{dtstring}')


known_face=findEncoding(images)

cap=cv2.VideoCapture(0)
while True:
    success, img=cap.read()
    imgS=cv2.resize(img, (0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    

    faces_current=face_recognition.face_locations(imgS)
    encode_current=face_recognition.face_encodings(imgS, faces_current)

    for encodeFace, locFace in zip(encode_current,faces_current):
        matches=face_recognition.compare_faces(known_face,encodeFace)
        faceDis=face_recognition.face_distance(known_face,encodeFace)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=locFace
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255),2)
            markAttendence(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)