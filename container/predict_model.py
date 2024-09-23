import numpy as np
import onnxruntime as ort

import utils
import dataIO


class SymbolDetector: 

    def __init__(self, yolo_weights, cuda=False):
        self.symbol_counter = 0

        self.min_symbol_distance = 10 # Pixel

        self.cuda = cuda
        self._setup(yolo_weights)

    def _setup(self, yolo_weights):

        if self.cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(yolo_weights, providers=providers)


    def predict(self, image):

        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]:image}

        # Run inference
        outputs = self.session.run(outname, inp)[0]
        
        return outputs
    

    def predict_symbols(self, data, image_uuid, img, tiling_size, tile_overlap, tiling_scale, iou_threshold, confidence_threshold, SYMBOL_ID=0):

        # image_path = data["images"][image_uuid]["filepath"]

        # # Read image
        # image = cv2.imread(image_path)
        image = img

        # Prepare image for detection and slice into tiles
        image_prepared = dataIO.prepareImage(image)
        tiles, anchors = dataIO.tileImage(image_prepared, tiling_size=tiling_size, overlap=tile_overlap, scale=tiling_scale)

        # All predictions are first merged into a list
        predictions = []
        i = 0
        for tile_image, (anchor_dx, anchor_dy) in zip(tiles, anchors): # Iteratre over all tiles

            # Convert to work with YOLOv7
            tile_image = tile_image.transpose((2, 0, 1))
            results = self.predict(tile_image) # Predict where symbols might be. :D

            if i % 50==0:
                print(i, len(tiles))

                #TODO
                #break
            i += 1
            for r in results:
                batch_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, cls_id, conf = r

                # Ignore uncertain predictions
                if conf < confidence_threshold:
                    continue

                # Adjust scale and translation and shift box into global image context
                x1, x2 = bbox_x1/tiling_scale+anchor_dx, bbox_x2/tiling_scale+anchor_dx
                y1, y2 = bbox_y1/tiling_scale+anchor_dy, bbox_y2/tiling_scale+anchor_dy

                # Append to list
                predictions.append([batch_id, x1, y1, x2, y2, cls_id, conf])

        # Since tiles might overlap
        predictions = utils.non_maximum_suppresion(predictions, iou_threshold) 

        # Since symbols are rotation ambiguous, we also check that bbox center has a certain distance
        predictions = utils.min_center_distance(predictions, self.min_symbol_distance)

        # Export predictions in COCO format
        for pred in predictions:
            batch_id, x1, y1, x2, y2, cls_id, confidence = pred

            data["symbols"][utils.genUUID()] = {   
                    "id": self.symbol_counter, 
                    "image_uuid": image_uuid, 
                    "bbox": [x1, y1, x2, y2], 
                    "keypoints": [[(x1+x2)/2, (y1+y2)/2]],
                    "connections": {
                    "0": -1, # left side
                    "1": -1, # right side
                    "2": -1, # top side
                    "3": -1, # down side              
                    }
                }
            self.symbol_counter += 1

        #print(f"Done looking for symbols! I found {len(predictions)} symbols.")
    
        return data
        

