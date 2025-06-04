# import cv2

# image = cv2.imread("Ishant.jpg")

# classNames = []
# classFile = 'config_files/coco.names'

# with open(classFile, 'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n') 

# configPath = 'config_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weightsPath = 'config_files/frozen_inference_graph.pb'

# net = cv2.dnn_DetectionModel(weightsPath,configPath)
# net.setInputSize(320,320)
# net.setInputScale(1.0/127.5)
# net.setInputMean((127.5,127.5,127.5))
# net.setInputSwapRB(True)

# classIds, confs, bbox = net.detect(image, confThreshold = 0.5) #if 50% confident, predict
# print(classIds, confs,bbox) 


# for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
#     cv2.rectangle(image,box,color=(0,255,0), thickness=2)
#     cv2.putText(image,classNames[classId-1],(box[0]+10,box[1]+30), 
#                 cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

# #Open
# cv2.imshow("Output", image)
# cv2.waitKey(0) 


import cv2 as c
import face_recognition as f


cap = c.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = c.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = f.face_locations(rgb_small_frame)
    face_encodings = f.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        c.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    c.imshow('Video', frame)
    
    if c.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
c.destroyAllWindows()