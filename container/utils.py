import imgaug.augmentables.bbs as bbs
import uuid

import math

def non_maximum_suppresion(list_prediction, iou_threshold=0.7):

    list_predictions_sorted = list_prediction.copy()
    list_predictions_sorted.sort(key=lambda x: x[6], reverse=True)


    list_predictions_reduced = []


    while len(list_predictions_sorted) > 0:
        prediction = list_predictions_sorted.pop(0) 
        list_predictions_reduced.append(prediction)
        
        batch_id, x1, y1, x2, y2, cls_id, confidence = prediction
        bbox_prediction = bbs.BoundingBox(x1, y1, x2, y2)

        list_predictions_sorted_copy = list_predictions_sorted.copy()
        for other_prediction in list_predictions_sorted:
            batch_id, x1, y1, x2, y2, cls_id, confidence = other_prediction
            bbox_other_prediction = bbs.BoundingBox(x1, y1, x2, y2)

            iou = bbox_prediction.iou(bbox_other_prediction)

            if iou > iou_threshold:
                list_predictions_sorted_copy.remove(other_prediction)
               
        list_predictions_sorted = list_predictions_sorted_copy


    return list_predictions_reduced


def min_center_distance(list_prediction, min_dist = 10):

    list_predictions_sorted = list_prediction.copy()
    list_predictions_sorted.sort(key=lambda x: x[6], reverse=True)


    list_predictions_reduced = []

    while len(list_predictions_sorted) > 0:
        prediction = list_predictions_sorted.pop(0) 
        list_predictions_reduced.append(prediction)
        
        batch_id, x1, y1, x2, y2, cls_id, confidence = prediction
        center0_x, center0_y = (x2+x1)/2, (y2+y1)/2

        list_predictions_sorted_copy = list_predictions_sorted.copy()
        for other_prediction in list_predictions_sorted:
            batch_id, x1, y1, x2, y2, cls_id, confidence = other_prediction
           
            center1_x, center1_y = (x2+x1)/2, (y2+y1)/2

            distance = math.pow(center1_x-center0_x, 2) + math.pow(center1_y-center0_y, 2)
            
            if distance < math.pow(min_dist, 2):
                list_predictions_sorted_copy.remove(other_prediction)
               
        list_predictions_sorted = list_predictions_sorted_copy


    return list_predictions_reduced


def genUUID():
    return str(uuid.uuid4())


def returnNotMe(my_list, item):
   
    my_list = [i for i in my_list if i != item]
    return my_list
