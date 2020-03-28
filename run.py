from service.image_processing import GLCM
from service import common
import cv2
import ntpath
from pathlib import Path

path_data_training = ''
path_data_testing = ''
path_export_data = str(Path(__file__).parent)+'/data/'

imnames = common.read_image(path_data_testing)

properties = []
properties.append(['Image name', 'Contrast', 'Dissimiliarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM', 'Target'])

# start pre processing using glcm
for imname in imnames:
    image = cv2.imread(imname)
    imname = ntpath.basename(imname)
    properties.append(GLCM.preprocess_glcm(image, imname))

# export data
common.export_data(path_export_data+'testing_dataset.csv', properties)