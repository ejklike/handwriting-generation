import os
import pickle
import random
import svgwrite
import xml.etree.ElementTree as ET

import numpy as np
from IPython.display import SVG, display

def _get_bounds(data):
    # get all x, y coordinates
    data = np.cumsum(data, axis=0)
    x, y = data[:, 0], data[:, 1]

    # min and max
    min_x, min_y = np.min(x), np.min(y)
    max_x, max_y = np.max(x), np.max(y)
  
    return min_x, max_x, min_y, max_y

# old version, where each path is entire stroke (smaller svg size, but have to keep same color)
def draw_strokes(data, factor=10, svg_fname='sample.svg', display=False):

    dtype = np.float16

    data = np.array(data, dtype=dtype)
    data[:, :2] /= factor

    margin_x, margin_y = 25, 25

    min_x, max_x, min_y, max_y = _get_bounds(data)
    dims = (
        2 * margin_x + max_x - min_x, 
        2 * margin_y + max_y - min_y)

    dwg = svgwrite.Drawing(svg_fname, size=dims)
    dwg.add(
        dwg.rect(insert=(0, 0), size=dims, fill='white'))

    base_str = '{}{},{} '

    # starting point (lefttop)
    command, pen_lifted = 'M', 1
    start_x, start_y = margin_x - min_x, margin_x - min_y
    p = base_str.format(command, start_x, start_y)

    # SVG commands
    # https://www.w3.org/TR/SVG/paths.html
    #
    # moveto: M(absolute), m(relative)
    # lineto: L(absolute), l(relative)

    for i in range(len(data)):
        if pen_lifted == 1:
            command = 'm'
        else:
            command = 'l' if command != 'l' else ''
        dx, dy, pen_lifted = data[i]
        p += base_str.format(command, dx, dy)

    color = 'black'
    width = 1

    dwg.add(
        dwg.path(p).stroke(color, width).fill('none'))
    dwg.save()

    if display:
        display(SVG(dwg.tostring()))


class DataLoader(object):

    def __init__(self): 
        self.data_dir = './data'
        raw_data_dir = os.path.join(
            self.data_dir, 'lineStrokes')
        data_pkl = os.path.join( # preprocessed data file
            self.data_dir, 'strokes_training_data.pkl')

        # check whether data is preprocessed
        if not os.path.exists(data_pkl):
            print('I Creating training data pkl file from raw data...')
            self._preprocess_to_pkl(raw_data_dir, data_pkl)
        
        # load data
        self._load_data(data_pkl)

    def _preprocess_to_pkl(self, raw_data_dir, data_pkl):
        # load all filenames
        file_list = []
        for root, dirs, files in os.walk(raw_data_dir):
            file_list += [os.path.join(root, f) for f in files]
        
        def _parse_xml(fname):
            """ parse a xml file into a np array of strokes
            """
            print('I Parsing {}'.format(fname), flush=True, end='\r')
            
            # parse xml file to a list of (x, y)
            tree = ET.parse(fname)
            root = tree.getroot()

            # description and stroke points
            desc, stroke_xml = root

            # offset (minimum of x and y coordinates)
            margin = 100
            x_offset = float(desc[2].attrib['x']) - margin
            y_offset = float(desc[3].attrib['y']) - margin

            # a stroke (xml) to point tuples
            def _get_points(stroke):
                """
                input: xml of a stroke
                output: a list of [x, y, end_of_stroke]
                    - end_of_stroke: the only last value is 1
                """
                x = lambda point: float(point.attrib['x']) - x_offset
                y = lambda point: float(point.attrib['y']) - y_offset
                point_list = [[x(point), y(point), 0] for point in stroke]
                point_list[-1][2] = 1
                return point_list
            
            stroke_list = [_get_points(s) for s in stroke_xml]
            flatten = lambda l: [item for sublist in l for item in sublist]
            point_list = flatten(stroke_list)
            
            def _get_np_array(points):
                """
                input: a list of (x, y, eos)
                output: a numpy array [delta_x, delta_y, end_of_stroke]
                """
                dtype = np.int16
                
                # get a diff array: delta (x, y), starting point: (0, 0)
                stroke_array = np.array(points, dtype=dtype)
                stroke_array[:, :2] = np.diff(
                    np.vstack((np.zeros((1,2)), stroke_array[:, :2])), axis=0)
                return stroke_array
            
            return _get_np_array(point_list)

        strokes = [_parse_xml(fname) for fname in file_list]

        # dump data
        with open(data_pkl, 'wb') as fout:
            pickle.dump(strokes, fout)

    def _load_data(self, data_pkl):
        # load preprocessed data
        with open(data_pkl, 'rb') as fin:
            self.raw_data = pickle.load(fin)
        print('I Loading data... len(data) = {}'.format(len(self.raw_data)))

    def pipeline(self, scale_factor=10, seq_length=300, batch_size=100):
        # goes thru the list, and only keeps the text entries that have more than seq_length points
        self.data = []
        self.valid_data =[]
        counter = 0

        data[:, :2] /= self.scale_factor

    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            data = self.valid_data[i % len(self.valid_data)]
            x_batch.append(np.copy(data[0:self.seq_length]))
            y_batch.append(np.copy(data[1:self.seq_length+1]))
        return x_batch, y_batch

    def reset_batch_pointer(self):
        pass