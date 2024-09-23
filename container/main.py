
import logging

logger = logging.Logger("MyLogger")

import SymbolConnection, ConnectionClassification, MeasureRecognition, ScaleVoting
import predict_model as YOLOv7
import utils

import numpy as np
from pathlib import Path

import cv2
import json
import os
import zipfile

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        


def crop_image(image, bbox):
    """
    Crop an image to the specified bounding box.

    Parameters:
    - image: A numpy array representing the image. Expected shape is (height, width, channels).
    - bbox: A tuple or list of four integers specifying the bounding box in the format (x, y, width, height),
            where (x, y) is the top-left corner, and 'width' and 'height' are the dimensions of the bounding box.

    Returns:
    - A cropped numpy array of the image within the bounding box.

    Raises:
    - ValueError: If bbox dimensions are invalid or exceed the image dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    
    if len(bbox) != 4:
        raise ValueError("The bbox must be a tuple or list of four elements.")
    
    img_h, img_w, img_c = image.shape
    x, y, w, h = bbox

    x0 = max(x, 0)
    y0 = max(y, 0)

    x1 = min(x + w, img_w)
    y1 = min(y + h, img_h)


    cropped_image = image[y0:y1, x0:x1]
    return cropped_image        

def load_json(file_path):

    if not Path(file_path).exists:
        raise FileNotFoundError(file_path)

    with open(file_path, "r") as fr:
        data = json.load(fr)
    return data

def load_drawing(file_path):
    if not Path(file_path).exists:
        raise FileNotFoundError(file_path)
    
    return cv2.imread(str(file_path))


def load_json(file_path):

    if not Path(file_path).exists:
        raise FileNotFoundError(file_path)

    with open(file_path, "r") as fr:
        data = json.load(fr)
    return data


def cast_file_type(file, type):
    if isinstance(file, type):
        return file
    else:
        return type(file)

def check_file_type(file, type=".json"):
    cast_file = cast_file_type(file=file, type=Path)

    if cast_file.suffix.lower() == type:
        return True
    else:
        return False



def load_main_json(input_files, filename):
    for file in input_files:
        if check_file_type(file=file, type=".json"):
            p = Path(file)
           
            if filename.lower() in p.name.lower():
                json_data = load_json(file)
                
                return json_data, file
        
    raise FileNotFoundError(f"No JSON file found in the provided list of files {input_files}.")
        


def infer_drawing_scale(input_paths, output_dir: str):

    input_dir = Path(Path(input_paths[0]).parent)
    

    with open("config.json", "r") as fr:
        config = json.load(fr)

    if not type(input_paths) == list:
        input_paths = [input_paths]
        

    # Filter json from images to accommodate differences between infra and building demonstrators
    IS_INFRA = False
    
    input_path = []
    for p in input_paths:

        p = Path(p)
        datatype = p.suffix.lower()
       
        if datatype == ".png":
            input_path.append(p)
        elif datatype == ".jpg":
            input_path.append(p)
        elif datatype == ".json":
            pass
        elif datatype == ".zip":
            IS_INFRA = True
            input_path = p
            break
        else:
            raise TypeError(f"The input data type {datatype} is not allowed")


    # Initiate Symbol Detector
    SymbolDetector = YOLOv7.SymbolDetector(config["yolo_weights_path"])

    # Initiate Symbol Detector
    SymbolConnector = SymbolConnection.SymbolConnector()

    # Initiate Symbol Detector
    ChainDetector = ConnectionClassification.ConnectionClassifier()
    
    # Initiate Symbol Detector
    MeasureRecognizer = MeasureRecognition.MeasureRecognizer(config["model_storage_directory"], config["user_network_directory"], config["download_weights"])

    # Initiate Symbol Detector
    ScaleVoter = ScaleVoting.ScaleVoter()
    
    


    scales = {}

    

    if not Path(output_dir).exists():
        Path.mkdir(Path(output_dir))

    #logger.warning("All modules are initialized. Service is started.")
    print("All modules are initialized. Service is started.")


    # If is infra, than we iterate over the drawings and views in the json
    # If it is building, we iterate over all png

    data = {}
    data["categories"] = {
            utils.genUUID():{
            "id": 0,
            "name": "Dimensional Symbol",
            "keypoints": ["center_point"],
            "skeleton": []
            }
        }
    
    # Init all dicts
    data["symbols"] = {}
    data["images"] = {}


    if not IS_INFRA:
        # Iterate over all files
        for i in range(len(input_path)): 
            file = input_path[i]

            # Does file exist?
            if not file.is_file():
                #logger.error(f"File at {str(file.name)} does not exist and is therefore ignored!")
                print(f"File at {str(file.name)} does not exist and is therefore ignored!")
                continue
                
           
            img = cv2.imread(str(file))

            
            img_uuid = utils.genUUID()
            h, w, c = img.shape
            

            data["images"][img_uuid] = {
                    "filepath": str(file),
                    "height": h,
                    "width": w,
                    "id": i
                }

            #logger.warning(f"Searching for dimension symbols.")
            print("Searching for dimension symbols.")
            data = SymbolDetector.predict_symbols(data, img_uuid, img, config["tiling_size"], config["tile_overlap"], config["tiling_scale"], config["iou_threshold"], config["confidence_threshold"])
        
            # Establish connections between the individual symbols  
            #logger.warning("Connecting detected symbols.")      
            print("Connecting detected symbols.")
            data = SymbolConnector.connect(img_uuid, img, data)
            
            
            # Classify connections   
            #logger.warning("Classifying the connections.")
            print("Classifying the connections.")
            data = ChainDetector.classify_connections(data)

        
            # Read measurements
            #logger.warning("Recognizing measurements.")
            print("Recognizing measurements.")
            data = MeasureRecognizer.read_measurements(img_uuid, img, data)


            # Infer scale
            #logger.warning("Inferring the scale.")
            print("Inferring the scale.")
            scale = ScaleVoter.inferScale(img_uuid, img, data)

            #logger.warning(f"Inferred scale {scale} meter per pixel.")
            print(f"Inferred scale {scale} meter per pixel for {file.name}.")


            scales[str(file.name)] = scale
            ScaleVoter.reset()
                
        out_path = Path.joinpath(Path(output_dir), "results.json")
        with open(str(out_path), "w") as fw:
            json.dump(scales, fw, cls=NpEncoder)



    else:

        if type(output_dir) == str:
            output_dir = Path(output_dir)

        input_file = input_path
        input_dir = input_file.parent
        input_folder = Path(input_dir, "input_files/")

        if input_file.suffix.lower() == ".zip":
            if input_folder.exists():
                _ = [os.remove(f) for f in input_folder.iterdir()]
            else:
                os.mkdir(str(input_folder))
                
            with zipfile.ZipFile(str(input_file), 'r') as zip_ref:
                zip_ref.extractall(str(input_folder))
        else:
            raise TypeError("Only ZIP files allowed")
        
        
        _ = [os.remove(f) for f in output_dir.iterdir()]
        
        input_filelist = []
        for f in input_folder.iterdir():
            input_filelist.append(str(f))

        drawing_data, drawing_data_filepath = load_main_json(input_filelist, "BIMKIT_Exchange")
                

        image_path_dict = {}
        for f in input_filelist:
            file = Path(f)

            if not file.suffix.lower() == ".png":
                continue

            filename = file.name
            image_path_dict[str(filename).lower()] = str(file)


        for drawing in drawing_data["drawings"]:
            drawing_id = drawing["ID"]
            drawing_filename = drawing["filename"]



            drawing_path = image_path_dict[str(drawing_filename).lower()]
            drawing_image = load_drawing(str(drawing_path))

            iterator_views = 0

            #logging.info("Load image")
            for view in drawing["views"]:
                view_id = view["view_id"]
                view_type = view["view_type"]
                view_position = view["position"]

                view_image = crop_image(drawing_image, view_position)
                view_img_height, view_img_width, img_channels = view_image.shape


                img_uuid = utils.genUUID()
                h, w, c = view_image.shape
                

                data["images"][img_uuid] = {
                        "filepath": str(file),
                        "height": h,
                        "width": w,
                        "id": iterator_views
                    }

                #logger.warning(f"Searching for dimension symbols.")
                print("Searching for dimension symbols.")
                data = SymbolDetector.predict_symbols(data, img_uuid, view_image, config["tiling_size"], config["tile_overlap"], config["tiling_scale"], config["iou_threshold"], config["confidence_threshold"])
            
                # Establish connections between the individual symbols  
                #logger.warning("Connecting detected symbols.")      
                print("Connecting detected symbols.")
                data = SymbolConnector.connect(img_uuid, view_image, data)
                
                
                # Classify connections   
                #logger.warning("Classifying the connections.")
                print("Classifying the connections.")
                data = ChainDetector.classify_connections(data)

            
                # Read measurements
                #logger.warning("Recognizing measurements.")
                print("Recognizing measurements.")
                data = MeasureRecognizer.read_measurements(img_uuid, view_image, data)


                # Infer scale
                #logger.warning("Inferring the scale.")
                print("Inferring the scale.")
                scale = ScaleVoter.inferScale(img_uuid, view_image, data)

                #logger.warning(f"Inferred scale {scale} meter per pixel.")
                print(f"Inferred scale {scale} meter per pixel.")

                iterator_views += 1
                view["service_133"] = round(scale, 5)


        output_dir = Path(output_dir)

        output_filename = Path(drawing_data_filepath).name
        output_final = Path.joinpath(output_dir, str(output_filename))

        with open(str(output_final), "w") as fw:
            json.dump(drawing_data, fw, cls=NpEncoder)


        output_files = [str(f) for f in output_dir.iterdir()]

        output_zipfile = Path.joinpath(output_dir, "output_service_133.zip")


        with zipfile.ZipFile(str(output_zipfile), 'w') as zip_wrt:
            for f in output_files:
                zip_wrt.write(str(f), arcname=Path(f).name, compress_type=zipfile.ZIP_DEFLATED)
            
        

        _ = [os.remove(f) for f in output_dir.iterdir() if f.suffix.lower() != ".zip"]
   

    

import argparse

if __name__ == '__main__': 


    parser = argparse.ArgumentParser(description="Runs Services")
    parser.add_argument("-i","--inputfile", nargs='+', default=[], help="Path to input data")
    parser.add_argument("-o","--outputPath", type=str, default="", help="Path to output directory")

    args = parser.parse_args()
    inputfile = args.inputfile
    outputpath = args.outputPath 
        
    infer_drawing_scale(inputfile, outputpath)




    


