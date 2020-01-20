import cv2
import os
import sys
import numpy as np

#This function is for Haar-cascade: A Face detection algorithm.
def faceDetection(image):
 gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)	#Image grayscaling.
 faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)	

 return faces,gray

#This function returns labels in accordance to .yml file syntax. 
def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")	#Skipping files that startwith .
                continue

            id=os.path.basename(path)	#fetching subdirectory names
            img_path=os.path.join(path,filename)	#Joining image path to subdirectory
            print("img_path:",img_path)
            print("id:",id)
            image=cv2.imread(img_path)	#loading each image one by one
            if image is None:
                print("Image not loaded properly")
                continue
            faces_rect,gray=faceDetection(image)	#Calling faceDetection function to return faces detected in particular image
            if len(faces_rect)!=1:
               continue 	#Each class with images are being fed to classifier
            (x,y,w,h) = faces_rect[0]
            roi_gray = gray[y:y+w,x:x+h]	#cropping region of interest 
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


#This function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces,faceID):
#    print(help(cv2.face))
    model = cv2.face.LBPHFaceRecognizer_create()
#    model = cv2.face.EigenFaceRecognizer_create()
    model.train(faces,np.array(faceID))
    return model

#This function draws bounding boxes around detected face in image
def draw_rect(image,face):
    (x,y,w,h)=face
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),thickness=2)

#This function writes name of person for detected label
def put_text(image,text,x,y):
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

