import numpy as np
import cv2

import shapely.geometry 

import utils

# 1. Runde
# Wir nehmen an, dass alle Symbole erkannt wurden
#
# Alle Symbole sind nicht überprüft (not fixed)
# Start in alle 4 Richtungen je Symbol (entlang des Ausrichtungswinkels), müssen aber globaler Natur angegeben werden. Wichtig für später. 
# Hier muss dann in Zukunft die weitere Suche nach Symbolen stattfinden.
# Gefundene Verbindungen müssen je Richtung abgesichert werden. Jedes Symbol kennt nur seine Nachbarn.

# Suche in eine Richtung, bis einer der drei Fälle Auftritt:
## 1. Wir schneiden ein Symbol, d.h. unser Richtungsvektor schneidet einen Kreis mit einem gewissen Radius (ggf. Quadrat) um einen Symbolmittelpunkt
## 2. Die Linie endet, kommen nur noch weiße Pixel
## 3. Wir erreichen das Ende des Bildes

class SymbolConnector():
    def __init__(self) -> None:
        # Hyperparameters
        ## Intersection with other symbol
        self.min_distance_symbol = 5
        self.boundary = 20

        ## End of line
        self.no_more_line_to_follow = 30 # number of time steps

        ## Field of view
        self.FoV = 30 # field of vision in pixels, how wide we look
        self.RoV = 10 # range of vision in pixels, how far we look

        ## Moving position
        self.max_speed = 20 # Maximum to update position
        self.momentum = 200 # Number used to have a specific inertia to the roboter

        ## Steering vector
        self.steering_rate = 0.01
        self.max_steering_rate = self.max_speed/5

        
        # Image stuff
        self.threshold = 200
        self.max_pix_value = 255       


        EPSILON = 1e-10
        # General orientation
        # 0 --> left side 
        # 1 --> right side
        # 2 --> upper side
        # 3 --> down side
        self.direction_vector = [
            [-1, 0],
            [1, 0],
            [EPSILON, -1], # Epsilon, so arctan is not infinite
            [EPSILON, 1]
        ]

        self.opposite_sides = ["1", "0", "3", "2"] # Not nice but easy 

    def connect_symbol(self, current_symbol_uuid, data, all_symbols, all_symbols_in_image_keys, image_uuid, min_connection_length):
            


            temp_connections = {}
            temp_symbol_connections = {}



            current_symbol_info = all_symbols[current_symbol_uuid]
            
            list_of_other_symbols = {other_symbol_uuid: all_symbols[other_symbol_uuid]["keypoints"][0] for other_symbol_uuid in all_symbols_in_image_keys if other_symbol_uuid != current_symbol_uuid}

            position_vector = current_symbol_info["keypoints"][0]
                   
            # Iterate over all symbol sides
            for side, existing_connection_id in current_symbol_info["connections"].items():

                # Side is already connected to another symbol
                if existing_connection_id != -1:
                    continue
                
                # Translate to direction
                direction_vector = self.direction_vector[int(side)]

                # Search for other symbol in this direction
                connection_found, connected_symbol_id = self._follow_line(
                    position_vector, direction_vector, list_of_other_symbols, current_symbol_uuid)

                # We found another symbol
                if connection_found:
                    
                    pos_0 = np.array(data["symbols"][current_symbol_uuid]["keypoints"][0])
                    pos_1 = np.array(data["symbols"][connected_symbol_id]["keypoints"][0])

                    connection_length = np.linalg.norm((pos_1-pos_0), 2)

                    if connection_length > min_connection_length:

                        connection_uuid = utils.genUUID()
                        
                        # Update symbols
                        #data["symbols"][current_symbol_uuid]["connections"][side] = connection_uuid
            
                        
                        # The connection must be established at the detecte symbol,
                        # there is the opposite side to connect
                        opposite_side = self.opposite_sides[int(side)]
                        #data["symbols"][connected_symbol_id]["connections"][opposite_side] = connection_uuid


                        temp_symbol_connections[connection_uuid] = [[side, current_symbol_uuid], [opposite_side, connected_symbol_id]]

                        # Save image uuid
                        temp_connections[connection_uuid] = {"image_id": image_uuid}
                        
                        # Save new connection
                        temp_connections[connection_uuid] = {"symbol_uuids": [current_symbol_uuid, connected_symbol_id]}


            return temp_connections, temp_symbol_connections


    def connect(self, image_uuid, img, data, min_connection_length=10, draw_image=False):

        #image = cv2.imread(data["images"][image_uuid]["filepath"])

        image = img

        self.drawing_image = image.copy()
        self.draw_vectors = draw_image

        self.image_height, self.image_width, self.image_channels = image.shape

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, self.image_binary = cv2.threshold(img_gray, self.threshold, self.max_pix_value, cv2.THRESH_BINARY_INV)
       

        connections = {}

        all_symbols = data["symbols"]

    
        all_symbols_in_image_keys = [sym_uuid for sym_uuid in all_symbols.keys() if all_symbols[sym_uuid]["image_uuid"] == image_uuid]

        
        from multiprocessing import Pool
        import os
        NUM_AGENTS = 32#os.cpu_count
        CHUNKSIZE = 20
        # Initiate multiprocessing with max number of agents
        pool = Pool(NUM_AGENTS, maxtasksperchild=CHUNKSIZE)

        # Prepare distance calculations with kwargs
        from functools import partial
        func = partial(self.connect_symbol, data=data, all_symbols=all_symbols, all_symbols_in_image_keys=all_symbols_in_image_keys, image_uuid=image_uuid, min_connection_length=min_connection_length)

        temp_sym_con = {}
        
        for temp_con, temp_sym in pool.imap_unordered(func, all_symbols_in_image_keys):
            connections = connections | temp_con
            temp_sym_con = temp_sym_con | temp_sym

        
        for connection_uuid in temp_sym_con.keys():
            [side, current_symbol_uuid], [opposite_side, connected_symbol_id] = temp_sym_con[connection_uuid]

            data["symbols"][current_symbol_uuid]["connections"][side] = connection_uuid
            data["symbols"][connected_symbol_id]["connections"][opposite_side] = connection_uuid

        # for current_symbol_uuid in all_symbols_in_image_keys:
        #     self.connect_symbol(data, connections, current_symbol_uuid, all_symbols, all_symbols_in_image_keys, image_uuid, min_connection_length)

        pool.close()
        pool.terminate()
                 

        #num_connections = len(list(connections.keys()))
        #print(f"Done with connecting symbols. I found {num_connections} connections!")
        
        data["connections"] = connections

        return data
        

    def _follow_line(self, starting_position, starting_direction, list_of_symbols_centers, my_symbol_uuid):
        self.list_of_symbols_centers = list_of_symbols_centers

        current_position_vector = np.array(starting_position, np.float64)
        current_velocity_vector = self._normalize_vector(starting_direction)
       
        only_white_pixel_counter = 0

        iter = 0
        while True:
            # First termination criterion when position exceed image
            if current_position_vector[0]+self.RoV >= self.image_width or current_position_vector[0]-self.RoV < 0:
                return False, -1
            if current_position_vector[1]+self.RoV >= self.image_height or current_position_vector[1]-self.RoV < 0:
                return False, -1
            
            current_steering_vector, only_white_pixel = self._single_time_step(current_position_vector, current_velocity_vector)
            

            # Second termination criterion when repeatedly encounter only white pixels, meaning line has ended
            if only_white_pixel:
                only_white_pixel_counter += 1
            else:         
                only_white_pixel_counter = 0 
           
            if only_white_pixel_counter > self.no_more_line_to_follow:
                return False, -1
            
            if self.draw_vectors and iter%5==0:
                color_steering_vector = (96,53,0) #BGR
                color_velocity_vector = (16,174,141)

                thickness_steering_vector = 1
                thickness_velocity_vector = 1 

                self.drawing_image = cv2.line(
                    self.drawing_image, (current_position_vector+current_velocity_vector).astype(np.uint16),
                    (current_position_vector+current_velocity_vector+current_steering_vector*self.steering_rate*50).astype(np.uint16),color_velocity_vector, thickness_velocity_vector)# Draw steering vector

                self.drawing_image = cv2.circle(self.drawing_image, (current_position_vector.astype(np.uint16)), 1, (255,0,0), thickness=-1) # Draw position vector
                self.drawing_image = cv2.line(self.drawing_image, current_position_vector.astype(np.uint16), (current_position_vector+current_velocity_vector).astype(np.uint64), color_steering_vector, thickness_steering_vector)# Draw velocity vector
                self.drawing_image = cv2.circle(self.drawing_image, current_position_vector.astype(np.uint16), 1, color_steering_vector, -1)
           

            # Update current velocity_vector by steering it
            try:
                if np.linalg.norm(current_steering_vector, ord=2) > self.max_steering_rate:
                    current_steering_vector = self._normalize_vector(current_steering_vector)*self.max_steering_rate
            except ValueError:
                print(current_steering_vector)
                print("Error found")
                #exit()

            current_velocity_vector = (current_velocity_vector*self.momentum + current_steering_vector*self.steering_rate)/self.momentum

            # if np.linalg.norm(current_velocity_vector, ord=2) > self.max_speed:
            #     current_velocity_vector = self._normalize_vector(current_velocity_vector)*self.max_speed


            # Third termination criterion when encounter another symbol
            encounter_symbol, symbol_idx = self._intersect_symbol(current_position_vector, current_velocity_vector, my_symbol_uuid)

            if encounter_symbol:
                return encounter_symbol, symbol_idx


            current_position_vector += current_velocity_vector

            iter += 1

        

    def _single_time_step(self, position_vector, velocity_vector):

        # Calcluate the pixel indecies for the view field
        velocity_angle = np.arctan(velocity_vector[1]/velocity_vector[0]) # arctan(y/x)

        absolute_velocity_x = position_vector[0] + np.cos(velocity_angle)*self.RoV
        absolute_velocity_y = position_vector[1] + np.sin(velocity_angle)*self.RoV

        # Calculate top and bottom corner of field of view
        field_of_view_top_x = absolute_velocity_x + np.sin(velocity_angle)*self.FoV/2
        field_of_view_top_y = absolute_velocity_y - np.cos(velocity_angle)*self.FoV/2  
        field_of_view_bottom_x = absolute_velocity_x - np.sin(velocity_angle)*self.FoV/2
        field_of_view_bottom_y = absolute_velocity_y + np.cos(velocity_angle)*self.FoV/2
        
        # Vector coordinates, that is along the field of view
        field_of_view_vector = np.array([field_of_view_bottom_x-field_of_view_top_x, field_of_view_bottom_y-field_of_view_top_y], np.float64)
       
   
        return self._calc_center_of_mass(field_of_view_vector, np.array([field_of_view_top_x, field_of_view_top_y]), np.array([absolute_velocity_x, absolute_velocity_y])) 



    def _calc_center_of_mass(self, field_of_view_vector, field_of_view_start, relative_center=np.zeros(2)):

        # Now we extract every pixel along the field of view
        field_of_view_length_rounded = int(np.linalg.norm(field_of_view_vector, ord=2))+1
        field_of_view_vector_normalized = self._normalize_vector(field_of_view_vector)

        field_of_view_pixel_x, field_of_view_pixel_y = int(field_of_view_start[0]), int(field_of_view_start[1])

        field_of_view_pixel_x_pred = field_of_view_start[0]
        field_of_view_pixel_y_pred = field_of_view_start[1]

        pixel_mass = 0
        center_of_mass = np.zeros(2, np.float64)
    
        for _ in range(field_of_view_length_rounded):
            field_of_view_pixel_x_pred = field_of_view_pixel_x_pred + field_of_view_vector_normalized[0]
            field_of_view_pixel_y_pred = field_of_view_pixel_y_pred + field_of_view_vector_normalized[1]

            if field_of_view_pixel_x == int(field_of_view_pixel_x_pred):
                if not field_of_view_pixel_y == int(field_of_view_pixel_y_pred):
                    field_of_view_pixel_y = field_of_view_pixel_y_pred
                else:
                    continue
            else:
                field_of_view_pixel_x = field_of_view_pixel_x_pred

                if not field_of_view_pixel_y == int(field_of_view_pixel_y_pred):
                    field_of_view_pixel_y = field_of_view_pixel_y_pred

            try:
                pixel_value = self.image_binary[int(field_of_view_pixel_y), int(field_of_view_pixel_x)]
            except IndexError:
                pixel_value = 0

            # Update pixel mass
            pixel_mass = pixel_mass + pixel_value

            # Update center_of_mass
            center_of_mass = center_of_mass + np.array([field_of_view_pixel_x-relative_center[0], field_of_view_pixel_y-relative_center[1]])*pixel_value

        # Somehow we must break the loop if we never see black pixel again
        if pixel_mass == 0:
            return np.array([0, 0]), True # If all pixels are white, overwrite center of mass to be in direction of velocity
        else:
            return center_of_mass/pixel_mass, False


    def _intersect_symbol(self, position_vector, velocity_vector, my_symbol_id):
        
        starting_point = position_vector
        ending_point = position_vector+velocity_vector

        # Instanciate movement as line
        motion_line = shapely.geometry.LineString([starting_point, ending_point])


        for symbol_guid, center_point in self.list_of_symbols_centers.items():


            if symbol_guid == my_symbol_id:
                continue

            x, y = center_point
            center_point_symbol = shapely.geometry.Point(x, y)

            circle_around_symbol = center_point_symbol.buffer(self.boundary).boundary

            #self.drawing_image = cv2.circle(self.drawing_image, (int(x), int(y)), 5, (128,255,150), -1)


            distance = shapely.shortest_line(center_point_symbol, motion_line).length

            if circle_around_symbol.intersects(motion_line):
                if distance > self.min_distance_symbol:
                    
                    return True, symbol_guid

        return False, -1
            

    def _normalize_vector(self, vector, order=2):

        if type(vector) is not np.array:
            vector = np.array(vector, np.float64)

        return vector/np.linalg.norm(vector, ord=order)
