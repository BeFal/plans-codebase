# import sympy and Point, Line
from shapely.geometry import LineString, Point

import numpy as np
import cv2


class ScaleVoter():
    def __init__(self):

        # Problem: Manche Zahlen sind in cm manche in m, wie kann ich den faktor 100 berücksichtigen?
        # Idee: Ich nehmen erstmal jede Zahl, wie sie ist. Also 1.00 und 50 interpretiere ich erstmal als m
        # 1 ist jetzt sehr klein im Vergleich zu 50, bildet also in einen sehr große Wert. 1.00m bildet den richtigen wert
        # Dann teilt man beides durch 100. 0,5 bildet jetzt den richtigen wert. Sehr nah an dem was 1,00 vorher hatte.
        # 0,01 (1/100) ergibt jetzt eine sehr große wert, landet am besten in einem sehr großen BIN oder außerhalb
        # Wenn ich jetzt sehr große und sehr kleine werte vernachlässige und bspw. nach sowas wie ner Gauss sache suche, die extreme ignoriert, passt es doch 
       
        # Maximum allowed scale
        self.max_scale = 1000

        # Number of bins from 0 to max_scale
        self.num_bins = 500

        # This number of bins will be ignore at the ends
        self.ignore_edge_bins = 1

        self.bin_size = self.max_scale/self.num_bins

        self.votes = np.zeros(self.num_bins, np.uint8)


    def reset(self):
        self.votes = np.zeros(self.num_bins, np.uint8)
        

    def inferScale(self, img_uuid, img, data, nearest_center=True):


        self.image = img
        # Get all connections
        connections = data["connections"]


        for connection_uuid, connection_info in connections.items():

            if "measurement" not in connection_info:
                continue

            # Get connected symbols
            connected_symbol0_uuid, connected_symbol1_uuid = connection_info["symbol_uuids"]

            symbol0_x1, symbol0_y1, symbol0_x2, symbol0_y2 = data["symbols"][connected_symbol0_uuid]["bbox"]
            p1 = ((symbol0_x2 + symbol0_x1)/2, (symbol0_y2 + symbol0_y1)/2)
          
            symbol_x1, symbol_y1, symbol1_x2, symbol_y2 = data["symbols"][connected_symbol1_uuid]["bbox"]
            p2 = ((symbol1_x2 + symbol_x1)/2, (symbol_y2 + symbol_y1)/2)

            # Establish line between symbols
            line = LineString([p1, p2])

            # Get importan line parameters such as center and length
            line_center = line.line_interpolate_point(0.5, True)
            line_length = line.length
        
            min_distance = line_length
            dimension_number = 0

            # Iterate over all measurements
            for dim, bbox, conf in connections[connection_uuid]["measurement"]:


                x1, y1, x2, y2 = bbox

                meas_x = (x1+x2)/2
                meas_y = (y1+y2)/2

                try:
                    dim = float(dim)
                except ValueError:
                    continue

                if dim <= 0:
                    continue

                measurement_center = Point(meas_x, meas_y)

                if nearest_center:
                    distance_val = line_center.distance(measurement_center) 
                else:
                    distance_val = line.distance(measurement_center)
                
                # If new measurement is closer, save dimension number
                if distance_val < min_distance:
                    min_distance = distance_val
                    dimension_number = dim
                


            
            if dimension_number == 0:
                continue
            

            # We calculate only in meter, see explanation at beginning of script
            dimension_meter_to_meter = dimension_number
            dimension_centimeter_to_meter = dimension_number/100
            dimension_millimeter_to_meter = dimension_number/1000


            scale_meter_to_meter = line_length/dimension_meter_to_meter
            scale_centimeter_to_meter = line_length/dimension_centimeter_to_meter
            scale_millimeter_to_meter = line_length/dimension_millimeter_to_meter
            

            bin_meter_to_meter = int(scale_meter_to_meter/self.bin_size)
            bin_centimeter_to_meter = int(scale_centimeter_to_meter/self.bin_size)
            bin_millimeter_to_meter = int(scale_millimeter_to_meter/self.bin_size)

            if bin_meter_to_meter <= self.num_bins:
                self.votes[bin_meter_to_meter] = self.votes[bin_meter_to_meter]+1
            
            if bin_centimeter_to_meter <= self.num_bins:
                self.votes[bin_centimeter_to_meter] = self.votes[bin_centimeter_to_meter]+1

            if bin_millimeter_to_meter <= self.num_bins:
                self.votes[bin_millimeter_to_meter] = self.votes[bin_millimeter_to_meter]+1


        
        return self.findScale()
    

    def findScale(self):

        #print(self.votes)

        max_count_bin = np.argmax(self.votes[1:])

        scale = max_count_bin*self.bin_size

        area = self.votes[max_count_bin-10:max_count_bin+10]

        if scale == 0:
            s = 0
        else:
            s = 1/scale

        return s



            








