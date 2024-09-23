import cv2

from pathlib import Path

def drawSymbolBBoxes(draw_image_uuid, data):

    for image_uuid, image_info in data["images"].items():
        if draw_image_uuid == image_uuid:
            filepath = Path(image_info["filepath"])
            break

        
    if not filepath.exists:
        raise ValueError("Image does not exist")

    img = cv2.imread(str(filepath))

    for symbol_info in data["symbols"].values():
        if draw_image_uuid == symbol_info["image_uuid"]:
            x1, y1, x2, y2 = symbol_info["bbox"]

            c = (96,53,0) # Blau
            c = (16,174, 141) # Grün
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 4)

            #img = cv2.putText(img, "Dimension Symbol", (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (96,53,0), 1)
        
    return img


def drawSymbolCenter(draw_image_uuid, img, data):
    
    image = img

    for symbol_info in data["symbols"].values():
        if draw_image_uuid == symbol_info["image_uuid"]:
            x1, y1 = symbol_info["keypoints"][0]
            
            image = cv2.circle(image, (int(x1), int(y1)), 5, (16,174,141), -1)

            id = symbol_info["id"]
            image = cv2.putText(image, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 1.1, (16,174,141), 2)
        
    return image



def drawConnections(draw_image_uuid, img, data, color=""):

    img = img

    if color == "":
        change_color = True
    else:
        change_color = False

    for current_connection_guid, current_connection_info in data["connections"].items():

        symbol_uuid0, symbol_uuid1 = current_connection_info["symbol_uuids"]

        x1, y1 = data["symbols"][symbol_uuid0]["keypoints"][0]
        x2, y2 = data["symbols"][symbol_uuid1]["keypoints"][0]

        if change_color:
            import random
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3) 
        
    return img


def drawOrientation(draw_image_uuid, data, scale=50):

    import random
    import math

    filepath = ""

    for image_uuid, image_info in data["images"].items():
        if draw_image_uuid == image_uuid:
            filepath = Path(image_info["filepath"])
            break
   
    if filepath == "":
        raise ValueError("Image does not exist")

    img = cv2.imread(str(filepath))

   
    for symbol_info in data["symbols"].values():
        if draw_image_uuid == symbol_info["image_uuid"]:
            if "orientation_angle" in symbol_info:
                x1, y1 = symbol_info["keypoints"][0]
                angle = symbol_info["orientation_angle"]

                x2 = x1 + math.cos(angle)*scale
                y2 = y1 + math.sin(angle)*scale

                c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 3) 


    return img


def drawConnectionsTypes(draw_image_uuid, img, data):


    img = img

    import random

    for current_connection_guid, current_connection_info in data["connections"].items():

        if "class" in current_connection_info:
            if current_connection_info["class"] == "inter":
                c = (50,50,100)
            elif current_connection_info["class"] == "intra": # Intra
                c = (174, 156, 103)
        else:
            c = (0,0,0)


        symbol_uuid0, symbol_uuid1 = current_connection_info["symbol_uuids"]

        x1, y1 = data["symbols"][symbol_uuid0]["keypoints"][0]
        x2, y2 = data["symbols"][symbol_uuid1]["keypoints"][0]

        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 3) 
        
    return img


def drawMeasurements(draw_image_uuid, img, data):

    # for image_uuid, image_info in data["images"].items():
    #     if draw_image_uuid == image_uuid:
    #         filepath = Path(image_info["filepath"])
    #         break
   
    # if not filepath.exists:
    #     raise ValueError("Image does not exist")

    # img = cv2.imread(str(filepath))
    img = img


    for current_connection_uuid, current_connection_info in data["connections"].items():
        
        # Maybe we missed a text
        if "measurement" not in current_connection_info:
            continue

        measurements = current_connection_info["measurement"]

        for text, bbox, confidence in measurements:
            x1, y1, x2, y2 = bbox

            xc = (x1+x2)/2
            yc = (y1+y2)/2

            img = cv2.circle(img, (int(xc), int(yc)), 5, (16, 174, 141), -1)

            img = cv2.putText(img, str(text), (int(xc-50), int(yc)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (16,174,141), 1)

            #img = cv2.cv2.putText(img, str(confidence), (int(xc), int(yc+20)), cv2.FONT_HERSHEY_COMPLEX, 1.1, (16,174,141), 2)

    return img


def drawScale(draw_image_uuid, data):

    for image_uuid, image_info in data["images"].items():
        if draw_image_uuid == image_uuid:
            filepath = Path(image_info["filepath"])
            break
   
    if not filepath.exists:
        raise ValueError("Image does not exist")

    img = cv2.imread(str(filepath))


    for current_connection_uuid, current_connection_info in data["connections"].items():

        if "scale" not in current_connection_info:
            continue

        current_scale, pixel_length, current_dimension, x, y = current_connection_info["scale"]

        s = f"{str(pixel_length)} / {str(current_dimension)}"# = {str(current_scale)}"
        img = cv2.cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.9, (16,174,141), 1)

            

    return img



    




