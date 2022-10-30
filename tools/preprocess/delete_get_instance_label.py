import json
import numpy as np
from labelme import utils

lbl_names = {'_background_': 0,
                'Motor Bike': 1,
                'Bus': 2,
                'LMV': 3,
                'Auto': 4,
                'Bike': 5,
                'Pedestrian': 6,
                'Pesestrian': 6,
                'LCV': 7,
                'E-rickshaw': 8,
                'Tractor': 9,
                'Truck': 10,
                # some annotation noise
                'Moter Bike': 1,
                'Moterbike': 1,
                'MotoeBike': 1,
                'm': 1,
                'e-rikshwa': 8,
                'e-rickshaw': 8,
                'MotorBike': 1,
                'cycle': 5,
                'BUS': 2,
                'truck': 10,
                'MortorBike': 1,
                'Motor Bike': 1,
                'e_rickshaw': 8,
                'Pedestrain': 6,
                'e-rickshow': 8,
                'e-Rickshaw': 8,
                'e-ricshaw': 8,
                'MoterBike': 1,
                'Rickshow': 8,
                'eRickshaw': 8,
                'Truk': 10,
                'Trauck': 10,
                'Motor bike': 1,
                'MototBike': 1,
                'bus': 1,
                'MotorBIke': 1,
                'Motir Bike': 1
                }


def instance_label(json_file):
    # print(json_file)
    assert json_file.endswith('.json')
    
    data = json.load(open(json_file))

    img = utils.img_b64_to_arr(data['imageData'])
    all_shapes = data['shapes']
    new_shapes = []
    for shape in all_shapes:
        points = shape["points"]
        xy = [tuple(point) for point in points]
        if len(xy) > 2:
            new_shapes.append(shape)
    cls, ins = utils.shapes_to_label(img.shape, new_shapes, lbl_names)
    # print(np.unique(cls), np.unique(ins))
    
    return cls, ins