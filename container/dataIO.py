import cv2
import numpy as np

import utils

def loadImages(image_paths, data):

    images = []
    data["images"] = {}

    for i, p in enumerate(image_paths):
        img = cv2.imread(str(p))

        images.append(img)

        h, w, c = img.shape

        filename = p.name
        filepath = str(p)

        
        data["images"][utils.genUUID()] = {
                "filename": filename,
                "filepath": filepath,
                "height": h,
                "width": w,
                "id": i
            }
        

        
def prepareImage(image, binary_conversion=False):

    image_copy = image.copy()
    
    if binary_conversion:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        _, image_copy = cv2.threshold(image_copy, 127, 255, cv2.THRESH_BINARY)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32)/255.0
    
    return image_norm




def tileImage(image, tiling_size=(640,640), overlap=(0, 0), scale=1.0):

    if type(tiling_size) == float or type(tiling_size) == int:
        tiling_size = (tiling_size, tiling_size)

    if type(overlap) == float or type(overlap) == int:
        overlap = (overlap, overlap)


    tiles = []
    anchor_points = []

    img_height, img_width, img_channels = image.shape


    if img_width <= tiling_size[0] or img_height <= tiling_size[1]:
        dsize = int(min(tiling_size[0]*scale, tiling_size[1])*scale)
        return cv2.resize(image, (dsize, dsize), cv2.INTER_LINEAR)



    tiles_pos_x = np.arange(0, img_width-tiling_size[0]+1, tiling_size[0]-overlap[0])
    tiles_pos_y = np.arange(0, img_height-tiling_size[1]+1, tiling_size[1]-overlap[1])

    
    if tiles_pos_x[-1] < img_width-tiling_size[0]:
        tiles_pos_x = np.append(tiles_pos_x, img_width-tiling_size[0])

    if tiles_pos_y[-1] < img_height-tiling_size[1]:
            tiles_pos_y = np.append(tiles_pos_y, img_height-tiling_size[1])

    x_tl, y_tl = np.meshgrid(tiles_pos_x, tiles_pos_y) # top_left corner of tiles


    for x, y in zip(x_tl.flatten(), y_tl.flatten()):

        tile = image[y:y+tiling_size[1], x:x+tiling_size[0]]

        dsize = int(min(tiling_size[0], tiling_size[1])*scale)
        tile = cv2.resize(tile, (dsize, dsize), interpolation=cv2.INTER_LINEAR)

        tiles.append(tile)
        anchor_points.append([x, y])




    return tiles, anchor_points





