""" 
Input : Filepath containing original image, predicted bounding boxes by Yolov5 model and depthmap predict by the network. 
Output : Image annotated with pedestrians and mean intensity of pixel in the depthmap.  
Usage : python3 asn1.py <path_to_input> <path_to_output>  
"""

import numpy as np
import cv2
import os
import syst

path_output  = sys.argv[1]
path_display = sys.argv[2]

list_img = []

#Finding the list of images
for r, d, f in os.walk(path_output):
    for file in f:
        if '.jpg' in file:
          	list_img.append(file)


for img_loc in list_img:
    
    #Correlating images with depthmap and annotations 
    loc = img_loc[:-4] + '.txt'
    file_output = path_output + loc
    img_path = path_output + img_loc
    img_path_out = path_display + img_loc
    df_path = img_path[:-4] + '_depthmap.png' 
    bbox_output = []
    depth_val   = []
    line_1 = []
    
    try:
        line_1 = open(file_output,"r").readlines()
    except FileNotFoundError:
        print("Wrong file or file path")
 
    for entry in line_1:
        bbox_1 = (entry.split(' ')[1:5])
        bbox_1 = [float(item) for item in bbox_1]
        bbox_output.append(bbox_1)
        
    #Adjusting bounding boxes to that of depth map size, finding mean intensity and nnotating the original image 
    frame = cv2.imread(img_path)
    df = cv2.imread(df_path)
    height,width = frame.shape[0],frame.shape[1]
    for (x,y,w,h) in bbox_output:
        left_1 =int((x - (w/2))*width)
        right_1 = int((x + (w/2))*width)
        bottom_1 = int((y + (h/2))*height)
        top_1 = int((y - (h/2))*height)
        y_ = frame.shape[0]
        x_ = frame.shape[1]
        t_y = df.shape[0]
        t_x = df.shape[1]
        x_scale = t_x/x_
        y_scale = t_y/y_
        left = int(np.round(left_1 * x_scale))
        top = int(np.round(top_1 * y_scale))
        right = int(np.round(right_1 * x_scale))
        bottom = int(np.round(bottom_1 * y_scale))
        c_depth = int(255 - np.mean(df[top:bottom,left:right]))
        cv2.rectangle(frame, (left_1, top_1), (right_1, bottom_1), (255,0,0),3)
        cv2.putText(frame,str(c_depth), (left_1, top_1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    cv2.imwrite(img_path_out,frame)
 
cv2.destroyAllWindows()
    
   
