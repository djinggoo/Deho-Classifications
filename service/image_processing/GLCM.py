import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import io

def preprocess_glcm(image, image_name):
    """Preprocessing for image using gray level coocurance matrix with distance 1 and angel 0.
    and get 5 properties. that is contrast, dissimiliarity, homogeneity, energy, correlation, ASM"""
    
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    result = greycomatrix(image_grayscale, [1], [0], 256, normed=True)
    target = '0' if image_name.startswith('TD') else '1'
    
    result = [image_name, 
              greycoprops(result, 'contrast')[0][0], 
              greycoprops(result, 'dissimilarity')[0][0], 
              greycoprops(result, 'homogeneity')[0][0],
              greycoprops(result, 'energy')[0][0],
              greycoprops(result, 'correlation')[0][0], 
              greycoprops(result, 'ASM')[0][0],
              target]
    return result


