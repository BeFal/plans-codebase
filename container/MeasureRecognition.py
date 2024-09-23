import cv2
import numpy as np

import easyocr
import math

from pathlib import Path

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="5"



# Search for the largest bounding box for a consecutive dimensional chain
# Extent this box with a specific margin to decrease errors
# Crop that image from the drawing
# Tile it?
# Use Easy OCR to detect Text
# Link each text box to the nearest dimensional chain
# Output format:
# Liste von Symbolen, Seite mit Guid zur Verbindung
# Liste von Verbindungen, Art der Verbindung Intra oder inter
 

class MeasureRecognizer():

    def __init__(self, model_storage_directory, user_network_directory, download_weights):

        
        self.reader = easyocr.Reader(['de'], gpu=True, model_storage_directory=model_storage_directory, user_network_directory=user_network_directory, download_enabled=download_weights)


        self.margin = 40
        self.text_confidence = 0.7

        
        # self.text_threshold = 0.7 # text confidence threshold
        # self.low_text = 0.5 # text low-bound score
        # self.link_threshold = 0.4 # link confidence threshold
        # self.tile_resize = 1024 # 'image size for inference
   

    
    def read_measurements(self, image_uuid, img, data):

        #img = cv2.imread(data["images"][image_uuid]["filepath"])
        img = img

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

        connections = data["connections"]

        for connection_uuid in connections.keys():
            if "class" not in data["connections"][connection_uuid]:
                continue

            connection_class = data["connections"][connection_uuid]["class"]
            
            if connection_class == "inter":
                continue

          
            connected_symbol_0_uuid, connected_symbol_1_uuid = data["connections"][connection_uuid]["symbol_uuids"]

            x1, y1, x2, y2 = data["symbols"][connected_symbol_0_uuid]["bbox"]
            symbol_0_x, symbol_0_y = (x2 + x1)/2, (y2 + y1)/2
      
            # Text box should be as big as the symbols are
            symbol_size = max(x2 - x1, y2 - y1)

            x1, y1, x2, y2 = data["symbols"][connected_symbol_1_uuid]["bbox"]
            symbol_1_x, symbol_1_y = (x2 + x1)/2, (y2 + y1)/2

            symbol_size = max(symbol_size, max(x2 - x1, y2 - y1))
            
            orientation_angle_chain = data["symbols"][connected_symbol_0_uuid]["orientation_angle"]

            x1, x2 = min(symbol_0_x, symbol_1_x), max(symbol_0_x, symbol_1_x)
            y1, y2 = min(symbol_0_y, symbol_1_y), max(symbol_0_y, symbol_1_y)

            tile_x1 = x1 - math.cos(orientation_angle_chain)*(self.margin+symbol_size)
            tile_x2 = x2 + math.cos(orientation_angle_chain)*(self.margin+symbol_size)
            
            tile_y1 = y1 - math.sin(orientation_angle_chain)*(self.margin+symbol_size)
            tile_y2 = y2 + math.sin(orientation_angle_chain)*(self.margin+symbol_size)

            if tile_x1 < 0:
                tile_x1 = 0
            if tile_x2 > img_bin.shape[1]:
                tile_x2 = img_bin.shape[1]


            if tile_y1 < 0:
                tile_y1 = 0            
            if tile_y2 > img_bin.shape[0]:
                tile_y2 = img_bin.shape[0]
 
            
            image_tile = img_bin[int(tile_y1):int(tile_y2), int(tile_x1):int(tile_x2)]
           

            rotated = False
            if orientation_angle_chain == 0:
                image_tile = cv2.rotate(image_tile, cv2.ROTATE_90_CLOCKWISE)
                rotated = True

            try: 
                if image_tile.size == 0:
                    results = []
                else:
                    try:
                        results = self._detect_text(image_tile)
                    except ValueError:
                        results = []
            except AttributeError:
                results = []
         
            measurement = []

            for res in results:

                bboxes, text, confidence = res


                if rotated:
                    bbox_y1, bbox_x1 = bboxes[0] 
                    bbox_y2, bbox_x2 = bboxes[2]
                    
                else:
                    bbox_x1, bbox_y1 = bboxes[0]
                    bbox_x2, bbox_y2 = bboxes[2]

                x1, x2 = bbox_x1+tile_x1, bbox_x2+tile_x1
                y1, y2 = bbox_y1+tile_y1, bbox_y2+tile_y1
                #xc, yc = (bbox_x2 + bbox_x1)/2+tile_x1, (bbox_y2 + bbox_y1)/2+tile_y1

                measurement.append([text, [x1, y1, x2, y2], confidence])
        
            data["connections"][connection_uuid]["measurement"] = measurement

        return data

           
           
        

    def _detect_text(self, image, rotation=[]):

        results = self.reader.readtext(image, allowlist="0123456789,.", text_threshold=self.text_confidence, link_threshold=0.3, rotation_info = rotation)
        
        return results
         
        
        



   