import numpy as np
import cv2 as cv
import os
import time

color_dict = {
    'bg': {'BGR' : [0., 0., 0.], 'key' : 0},
    'cup': {'BGR' : [0., 222., 0.], 'key' : 1},
    'plate': {'BGR' : [200., 35., 35.], 'key' : 2},
    'spoon' : {'BGR' : [36., 36., 157.], 'key' : 3},
    'cell-phone' : {'BGR' : [50., 190., 190.], 'key' : 4},
    'pouch' : {'BGR' : [190., 190., 50.], 'key' : 5},
    'beans' : {'BGR' : [0., 75., 110.], 'key' : 6}
    }

def mk_onehot_zeros (rgb_src:np.array) ->np.ndarray:
    # print(rgb_src.shape)
    h, w, ch = rgb_src.shape
    onehot_src = np.zeros((h*w,1), dtype=np.int64)
    return onehot_src
    
def use_np_where (rgb_src:np.array) -> np.ndarray:
    # print('use_np_where() called')
    time_onehot = time.time()
    onehot_src = mk_onehot_zeros(rgb_src)
    # print('\toneHot:', time.time()-time_onehot)

    time_reshape = time.time()
    h,w,ch = rgb_src.shape
    rgb_flatten = rgb_src.reshape(-1, 3)
    # print('\treShape :', time.time()-time_reshape)

    # time_unique = time.time()
    # print(len(np.unique(rgb_flatten, axis=0)))
    # print('\tunique :', time.time()-time_unique)

    for label, sub_dict in color_dict.items() :
        target_color = np.array(sub_dict['BGR'])
        onehot_color = sub_dict['key']
        # print(rgb_flatten==target_color.any())
        np.where((rgb_flatten==target_color).any(), 0, onehot_color)


def use_cv_inrange(rgb_src:np.array) -> np.ndarray:
    onehot_src = mk_onehot_zeros(rgb_src)  

    for label, sub_dict in color_dict.items() : 
        target_color = np.array(sub_dict['BGR'])
        onehot_color = sub_dict['key']
        mask = cv.inRange(rgb_src, target_color, target_color)
        cv.imshow('mask', mask)
        cv.waitKey(0)
        np.place(onehot_src, mask, onehot_color)
    
    cv.imwrite('./test_place.png', onehot_src.reshape(rgb_src.shape[:2]))




base_dir = './test/'
src = cv.imread(os.path.join(base_dir, 'lbl/free-coffee-samples.png'))

seq = 1

where_time_list = []
for _ in range(seq) :
    time1 = time.time()
    src_from_where = use_np_where(src)
    where_time_list.append(time.time()-time1)

print('where : ', sum(where_time_list)/seq)


place_time_list = []
for _ in range(seq) :
    time1 = time.time()
    use_cv_inrange(src)
    place_time_list.append(time.time()-time1)

print('place : ', sum(place_time_list)/seq)