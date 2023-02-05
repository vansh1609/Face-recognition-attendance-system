import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImagesAttendance'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImage=cv2.imread(f'{path}/{cl}')
    images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncoding(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encodeListknown=findEncoding(images)
print(len(encodeListknown))
print('Encoding complete')

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurrFrame=face_recognition.face_locations(imgS)
    encodingsCurrFrame=face_recognition.face_encodings(imgS,facesCurrFrame)

    for encodeFace,faceLoc in zip(encodingsCurrFrame,facesCurrFrame):
        matches=face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDist=face_recognition.face_distance(encodeListknown,encodeFace)
        print(faceDist)
        matchIndex=np.argmin(faceDist)


        if(matches[matchIndex]):
            name=classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
