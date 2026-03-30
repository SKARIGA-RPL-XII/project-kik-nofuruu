import pandas as pd 
import numpy as np 

class GestureEngine:
    def __init__(self, k=3):
        self.k = k 
        self.x_train = None 
        self.y_train = None 
        self.labels_map = {}
        self.data_path = pd.read_csv('data.csv')
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    