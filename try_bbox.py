import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import os

def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding,
                          previous_layer_rf_info):
    n_in = previous_layer_rf_info[0] # input size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]

def compute_proto_layer_rf_info_gu():
    rf_info = [224, 1, 1,1]
    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=3,
                                                layer_stride=1,
                                                layer_padding='SAME',
                                                previous_layer_rf_info=rf_info)
    return proto_layer_rf_info

def save_prototype_original_img_with_bbox(fname, color=(0, 255, 255)):
    info_arr = compute_proto_layer_rf_info_gu();
    bbox_height_start = int(info_arr[0])
    bbox_height_end = int(info_arr[1])
    bbox_width_start = int(info_arr[2])
    bbox_width_end = int(info_arr[3])
    p_img_bgr = cv2.imread(fname)
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname+'_bbox.jpg', p_img_rgb)

if __name__ == '__main__':
    root_path = os.getcwd() + '/dataset/try_box/'
    dirs = os.listdir(root_path)
    for temp_dir in dirs:
        save_prototype_original_img_with_bbox(root_path + temp_dir)