import time
import cv2
import mss
import numpy as np
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
title="number plate detection"
# ## Env setup
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils_self_created2 as vis_util

coordinate_list=[]
details=[]
# # Model preparation 
PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'labelmap.pbtxt'
NUM_CLASSES = 37


# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# # Detection
coordinates=[]
def image_detection():
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            base_path=''
            images=[]
            for base, _, image in os.walk('batch_test_files/'):
                base_path=base
                images=image
            for i in images:
                image=base_path+i
                # Get raw pixels from the screen, save it to a Numpy array
                image_np =cv2.imread(image)
                # To get real color we do this:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Visualization of the results of a detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1)

                #I am taking an example of text detection. to describe the usability of added lines of code.
                
                #as shown in the image attatched below, we have two lines written in the image, now to print the sequence as such, we have to have
                #the center coordinates of each character in the image.
                # code snippet to print and license number as such.
                #this is the main code which tells us the use of the arrays corresponding_coordinates and 
                corresponding_coordinates=np.array(vis_util.corresponding_coordinates)
                character=np.array(vis_util.label_array)
                #this is a 2d array which will contain all the characters and accuracies corresponding to them as a string.
                characters=[]
                for i in range(len(character)):
                    for string in character[i]:
                        characters.append(string.split(':')[0])
                #now i have got all the characters in a list named characters.
                threshold_y=np.mean(corresponding_coordinates[:, 1])
                #decided the threshold value of y coordinate. if any character coordinate lies above this threshold value
                #that will be printed first
                #and vice versa.
                #this threshold value is the mean of all the y coordinates of all the characters.
                characters=np.array(characters)
                sortable_initial_x_coordinates=corresponding_coordinates[:, 0][corresponding_coordinates[:, 1]<threshold_y]
                #got the coordinates for all the characters which are lying above the threshold value.
                starting_characters=characters[corresponding_coordinates[:, 1]<threshold_y][np.argsort(sortable_initial_x_coordinates)]
                #then printed the upper characters first in the sorted order of x values.
                sortable_final_x_coordinates=corresponding_coordinates[:, 0][corresponding_coordinates[:, 1]>=threshold_y]
                #repeated for the characters lying below the threshold value.
                ending_characters=characters[corresponding_coordinates[:, 1]>=threshold_y][np.argsort(sortable_final_x_coordinates)]
                #printed the ending characters which lie below the threshold y value, in the sorted order of x values.
                print(''.join([''.join(starting_characters), ''.join(ending_characters)]))
                vis_util.clearer()
                #this clearer function has to be called before switching to another image to clear two currently filled arrays.
                # namely, vis_util.corresponding_coordinates array and vis_utils.label_array.
            
            
            
image_detection()
