import numpy
import glob

def read_image(path):
    """Read data image from path. remember take all image in 1 folder"""
    folder = glob.glob(path)
    images = []

    for files in folder:
        for ifile in glob.glob(files):
            images.append(ifile)
    
    return images

def export_data(path, properties):

    """Export data properties to csv file with specific path"""
    numpy.savetxt(path, properties, fmt='%s', delimiter=',')
    return "data success to create"