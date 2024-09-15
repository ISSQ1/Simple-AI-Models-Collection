import cv2
"""
The OpenCV library was essential in the program for opening the camera,
capturing images, processing them (drawing boxes and adding text), 
displaying the results, and setting up and running the deep learning model for object detection.
OpenCV combines these tasks and provides the necessary tools to work with images and videos easily and effectively.

"""

# Open the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set width of the frame
cam.set(4, 480)  # Set height of the frame

# Read class names from coco.names file
classNames = []
classFile = 'coco.names'
with open(classFile , 'rt') as f:
     classNames = f.read().rstrip('\n').split('\n')
print(classNames)

# Load the model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start capturing and displaying video
while True:
     success, img = cam.read() # Capture frame-by-frame
     classIds, confs, bbox = net.detect(img, confThreshold=0.5)
     print(classIds)
     if len(classIds) > 0:
          for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
               cv2.rectangle(img, box, color=(0, 255, 0), thickness=2) # Draw rectangle around detected object
               cv2.putText(img, classNames[classId-1], (box[0]+10, box[1]+20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 0, 255), 2)
     cv2.imshow('output', img) # Display the resulting frame
     cv2.waitKey(1)

