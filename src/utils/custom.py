import numpy as np
from monai.transforms import Randomizable, Transform

class PoissonNoise(Transform, Randomizable):
    def __init__(self,
                 keys,
                 ratio=2, 
                 ps=None, 
                 ron=3, 
                 dk=7, 
                 ):
        self.keys=keys
        self.ratio = ratio
        self.ps = ps
        self.ron = ron
        self.dk = dk

    def randomize(self):
        pass

    def __call__(self, data):
        exp_time=float(data['exp_time'])
        for key in self.keys:
            img_data = data[key]  # expects a dictionary with an 'image' key containing a numpy array
            img_data = np.where(img_data < 0.00000000e+00, 0.00000000e+00, img_data)  # to not have minus pixels
            width, height = img_data.shape[1:3]

            img = img_data * exp_time  # e
            #TODO: Check absolute values /saw strong increase with noise adding of mean value
            DN = np.random.normal(0, np.sqrt(self.dk * exp_time / ((60 * 60) * self.ratio)), (width, height))
            RON = np.random.normal(0, self.ron, (width, height))
            SN = np.random.poisson(np.abs(img / self.ratio))  # this line is different

            noise_img = (SN + RON + DN) / (exp_time / self.ratio)

            noise_img = np.where(img_data == 0.00000000e+00, 0.00000000e+00, noise_img)

            data[key] = noise_img
        return data

class RemoveZeroImage(Transform):
    def __call__(self, data):
        if np.any(data['image'] <= 0.0):
            
            print("Delete!")
            return None
        else:
            print("Keep!")
            return data
        
class ReplaceValue(Transform):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = np.where(data[key] == 0.0, 0.0, data[key])
        return data