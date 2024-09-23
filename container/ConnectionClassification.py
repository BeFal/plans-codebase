import math
from pathlib import Path

import utils



EPSILON = 1e-10

class ConnectionClassifier():
    def __init__(self, num_levels=5) -> None:

        self.num_levels = num_levels # Might be necessary to detect inter_chains multiple times to spread certain orientation
        
        # General orientation
        # 0 --> left side 
        # 1 --> right side
        # 2 --> top side
        # 3 --> down side
        self.direction_vector = [
            [-1, 0],
            [1, 0],
            [EPSILON, -1], # Epsilon, so arctan is not infinite
            [EPSILON, 1]
        ]

        self.opposite_sides = ["1", "0", "3", "2"] # Not nice but easy

        
        self.verified_symbols = set()
        self.unverified_symbols = set()


        self.counter = [0,0,0]


    def classify_connections(self, data):
        data = self._classify_symbols(data)
        
        for _ in range(self.num_levels):
             data = self._verify_connections(data)
        

        # If there are any connections left and these connections are by them self, they are additionally classified as chains
        for current_symbol_uuid in self.unverified_symbols:
            current_symbol_connections = data["symbols"][current_symbol_uuid]["connections"]

            for dir, connected_symbol_uuid in current_symbol_connections.items():
                if connected_symbol_uuid == -1:
                    continue
               
                if dir == 0 or dir == self.opposite_sides[0]: # Meaning symbol is horizontal and connections are on opposite sides
                    self.verified_symbols.add(current_symbol_uuid)
                    data["symbols"][current_symbol_uuid]["orientation_angle"] =  math.pi/2
                else: # Meaning symbol is vertical and connections are on opposite sides
                    self.verified_symbols.add(current_symbol_uuid)
                    data["symbols"][current_symbol_uuid]["orientation_angle"] = 0
                break
      
        data = self._verify_connections(data)

        n, i, j = self.counter
        
        #print(f"Done with classifing the connections, there are {n} inter-connections, {i}intra-connections and {j} connections I cannot classify.")
        return data


    def _classify_symbols(self, data):

        for symbol_uuid, symbol_info in data["symbols"].items():
            
            number_of_positive_connections = 0

            for connection in symbol_info["connections"].values():
            # Check number of positive connections
                if connection != -1:
                    number_of_positive_connections += 1
                        
            if number_of_positive_connections == 2:
                if symbol_info["connections"]["0"] != -1 and symbol_info["connections"][self.opposite_sides[0]] != -1: # Meaning symbol is horizontal and connections are on opposite sides
                    self.verified_symbols.add(symbol_uuid)
                    data["symbols"][symbol_uuid]["orientation_angle"] =  math.pi/2
                elif symbol_info["connections"]["2"] != -1 and symbol_info["connections"][self.opposite_sides[2]] != -1: # Meaning symbol is vertical and connections are on opposite sides
                    self.verified_symbols.add(symbol_uuid)
                    data["symbols"][symbol_uuid]["orientation_angle"] = 0
                else:
                    self.unverified_symbols.add(symbol_uuid)
            elif number_of_positive_connections == 0:
                print("One symbol is all by himself. :'(")
            else:
                self.unverified_symbols.add(symbol_uuid)
   
        return data
    
   
    def _verify_connections(self, data):

        connections = data["connections"]
        
        new_verfied_symbols = set() # To add the need verified symbols to the overall list

        for current_symbol_uuid in self.verified_symbols: # Iterate over all certain symbols
            
            current_symbol_orientation = data["symbols"][current_symbol_uuid]["orientation_angle"]
            current_symbol_connections = data["symbols"][current_symbol_uuid]["connections"]

            # Identify sides in direction of orientation
            if current_symbol_orientation == 0:
                side_ids = ("2", self.opposite_sides[2])
            else:
                side_ids = ("0", self.opposite_sides[0])
                
               

            for orientation_id, connection_uuid in current_symbol_connections.items(): # Iterate over symbol connections
                if connection_uuid == -1:
                    continue
                
                try:
                    connected_symbol_uuid = utils.returnNotMe(connections[connection_uuid]["symbol_uuids"], current_symbol_uuid)[0] # Return the other symbol
                except KeyError:
                    continue
                                
                if  "orientation_angle" in data["symbols"][connected_symbol_uuid]:
                    connected_symbol_orientation = data["symbols"][connected_symbol_uuid][ "orientation_angle"]

                    if current_symbol_orientation == connected_symbol_orientation:
                        if orientation_id in side_ids: # Meaning the connection is perpendicular to the orientation
                            data["connections"][connection_uuid]["class"] = "intra" # therefore, it is an intra connection
                        else:
                            data["connections"][connection_uuid]["class"] = "inter"

                    else:
                        pass
    

                else:
                    data["symbols"][connected_symbol_uuid]["orientation_angle"] = current_symbol_orientation 
                    new_verfied_symbols.add(connected_symbol_uuid)
                   
                    if orientation_id in side_ids: # Meaning the connection is perpendicular to the orientation
                        data["connections"][connection_uuid]["class"] = "intra" # therefore, it is an intra connection
                    else:
                        data["connections"][connection_uuid]["class"] = "inter"
        
        self.verified_symbols.update(new_verfied_symbols)
        
        return data
    


    

       