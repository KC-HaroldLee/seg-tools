import cv2 as cv
import numpy as np
import json

color_dict = {
    'bg': {'BGR' : [0., 0., 0.], 'key' : 0},
    'cup': {'BGR' : [0., 222., 0.], 'key' : 1},
    'plate': {'BGR' : [200., 35., 35.], 'key' : 2},
    'spoon' : {'BGR' : [36., 36., 157.], 'key' : 3},
    'cell-phone' : {'BGR' : [50., 190., 190.], 'key' : 4},
    'pouch' : {'BGR' : [190., 190., 50.], 'key' : 5},
    'beans' : {'BGR' : [0., 75., 110.], 'key' : 6}
    }

with open('./test/json/free-coffee-samples.json', 'r') as json_file :
    json_data = json.load(json_file)

img_src = cv.imread('./test/ori/free-coffee-samples.jpg')

zero_src = np.zeros([json_data['imageHeight'], json_data['imageWidth'], 3], dtype=np.int8)

print(img_src.shape)
print(zero_src.shape)

for shape in json_data['shapes'] :
    color_value = color_dict[shape['label']]['BGR']
    points = np.array(shape['points'], dtype=np.int64)
    print(points)
    # points = np.array([[50,50],[100,50],[100,100],[50,100]], dtype=np.int64)
    zero_src = cv.drawContours(zero_src, [points], 0, tuple(color_value), -1, cv.LINE_8)
    cv.imshow('zero', zero_src)
    cv.waitKey(0)


cv.imwrite('./test/lbl/free-coffee-samples.png', zero_src)