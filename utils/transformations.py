from PIL import Image
import PIL.ImageOps    

class InvertColor(object):
    def __call__(self, img):
        return  PIL.ImageOps.invert(img)