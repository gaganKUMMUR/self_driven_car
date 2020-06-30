import cv2
import numpy as np
import os
import time
def diff(a, b):
    dif = a-b
    if dif > 0:
        return "overlap"
    if dif < 0:
        return "ok"
    else:
        return "ok"
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
# Load Yolo
net = cv2.dnn.readNet(weightsPath, configPath)
classes = []
with open("./yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape
    #(height,width)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

# Draw a diagonal blue line with thickness of 5 px
    cv2.line(frame,(int(width/2)-140,height-50),(int(width/2)-80,height-50),(0,0,255),2)
    cv2.line(frame,(int(width/2)+140,height-50),(int(width/2)+80,height-50),(0,0,255),2)
    cv2.line(frame,(int(width/2)+130,height-100),(int(width/2)-130,height-100),(0,0,255),2)
    cv2.line(frame,(int(width/2)-120,height-150),(int(width/2)-60,height-150),(50,255,255),2)
    cv2.line(frame,(int(width/2)+120,height-150),(int(width/2)+60,height-150),(50,255,255),2)
    cv2.line(frame,(int(width/2)-100,height-250),(int(width/2)-40,height-250),(0,255,255),2)
    cv2.line(frame,(int(width/2)+100,height-250),(int(width/2)+40,height-250),(0,255,255),2)
    cv2.line(frame,(int(width/2)-100,height-250),(int(width/2)-120,height-150),(0,255,0),2)
    cv2.line(frame,(int(width/2)+100,height-250),(int(width/2)+120,height-150),(0,255,0),2)
    cv2.line(frame,(int(width/2)-120,height-150),(int(width/2)-140,height-50),(255,0,0),2)
    cv2.line(frame,(int(width/2)+140,height-50),(int(width/2)+120,height-150),(255,0,0),2)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
	        #if int(x) < int(width/2):
		    #   cv2.putText(frame,"turn right",(10,10),font,3,color,3)
        print(boxes[0])		
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if int(height-100) > y + h > int(height-250):  
                if int(width/2)-120 < x < int(width/2)+120 and int(width/2)-120 < x + w <int(width/2)-120:
                    cv2.putText(frame,"stop",(300,40),font,3,color,3)
                if int(width/2)-120 < x + w < int(width/2)+120:
                    clear_bounding_box(x+w+50,y+h,x+w+50+260)
                    cv2.line(frame,(int(width/2)-140,height-50),(x+w+50,y+h),(0,0,255),2)
                    cv2.line(frame,(int(width/2)+140,height-50),(x+w+50+260,y+h),(0,0,255),2)
                    cv2.putText(frame,"turn right",(300,40),font,3,color,3)
                if int(width/2)-120 < x < int(width/2)+120:
                    clear_bounding_box(x-50,y+h,x-50-260)
                    cv2.line(frame,(int(width/2)-140,height-50),(x-50-260,y+h),(0,0,255),2)
                    cv2.line(frame,(int(width/2)+140,height-50),(x-50,y+h),(0,0,255),2)
                    cv2.putText(frame,"turn left",(300,40),font,3,color,3)
                
    def clear_bounding_box(x,y,width):
        cv2.line((frame),(x,y),(width,y),(0,0,255),2)
        
        return None
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
