import cv2
import os
import sys
import numpy as np
import train as fd

imagePath = sys.argv[1]		
image = cv2.imread(imagePath)

#image = cv2.imread('Test/4.jpg') 	#For PIP package
face_det,gray = fd.faceDetection(image)
print("faces detected",face_det)

#For Training
#faces,faceID = fd.labels_for_training_data('Train')
#model = fd.train_classifier(faces,faceID)
#model.save('trainData.yml')

name = {0:"Tom Cruise",1:"Christian Bale"}	# Adding names for each label

#For Testing
model=cv2.face.LBPHFaceRecognizer_create()
model.read('trainData.yml')	#Loads the previously trained data for Face Recognition.

for face in face_det:
    (x,y,w,h) = face
    roi_gray = gray[y:y+h,x:x+h]
    label,confidence = model.predict(roi_gray)	#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fd.draw_rect(image,face)
    predicted_name = name[label]
    if(confidence>51):	#To avoid repetition of labels, confidence is adjusted according to the Test dataset.
        continue
    fd.put_text(image,predicted_name,x,y)

cv2.imshow("Face Detection",image)
#cv2.imwrite("Face Detection",image)
cv2.waitKey(0)	# Press any key to exit 
cv2.destroyAllWindows

