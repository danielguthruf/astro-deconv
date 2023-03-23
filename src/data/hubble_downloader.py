# -*- coding: utf-8 -*-
import os
import gzip
import cv2
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from matplotlib import pyplot as plt
from astropy.visualization import LogStretch
from pathing import RAW_DATA_PATH,QUERY_PATH,PROCESSED_DATA_PATH
from astroquery.esa.hubble import ESAHubble
from astropy.io import fits


def findRotateImage(image):
    binary_mask=np.where(image>0.00000001,255,0).astype(np.uint8)

    cnts = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    rot_rect = cv2.minAreaRect(cnt)
    
    angle = rot_rect[2] # rect angle
    return ndimage.rotate(image, angle)


def crop(image):
    h,w = image.shape
    y_nonzero, x_nonzero= np.nonzero(image)
    ysen = int(np.maximum(np.min(y_nonzero),h-np.max(y_nonzero))+args.crop_margin*h)
    xsen = int(np.maximum(np.min(x_nonzero),w-np.max(x_nonzero))+args.crop_margin*w)
    return image[ysen:h-ysen,xsen:w-xsen]


def loadFitsGZ(filename):
    output_file= f"{filename}.fits"
    input_file=f"{output_file}.gz"
    #TODO: ADD support for multiple zipFiles, atm skip them
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(input_file)
    return fits.open(output_file)
def keepOnlyDataFits(hdul):

    # Loop over all the HDUs in the file
    if len(hdul)>1:
        for i, hdu in enumerate(hdul):
            
            # Check if this HDU contains the main image with optical data
            if hdu.header.get('EXTNAME') == 'SCI':
                new_hdul = fits.HDUList([hdul[0],hdu])
    else:
        new_hdul = hdul

    return new_hdul
def saveFits(hdul,filename):
    hdul.writeto(f"{filename}.fits.gz", overwrite=True)

def download(observation_id):
    try:
        filename=f"{sRAW_DATA_PATH}/{observation_id}"
        output_file= f"{filename}.fits"
        input_file=f"{output_file}.gz"
        hubbler.download_product(observation_id=observation_id, calibration_level=args.calibration_level,
                            filename=filename, product_type=args.intent)
        hdul = loadFitsGZ(filename)
        new_hdul = keepOnlyDataFits(hdul=hdul)
        # new_hdul = cropFits(hdul=new_hdul)
        if not args.prep:
            saveFits(hdul=new_hdul,filename=f"{filename}")
            hdul.close()
            new_hdul.close()
            if os.path.isfile(output_file): os.remove(output_file)
            return None
        else:
            if os.path.isfile(output_file): os.remove(output_file)
            if os.path.isfile(input_file): os.remove(input_file)
            return new_hdul
    except Exception as e:
        print("An error occurred:", e)
        print("Problematic Observation ID:", observation_id)
        if os.path.isfile(output_file): os.remove(output_file)
        if os.path.isfile(input_file): os.remove(input_file)
        return None

def preprocess(hdul,observation_id):
    i = 1 if len(hdul) > 1 else 0
    image = hdul[i].data
    rotated_image = findRotateImage(image)
    cropped_image = crop(rotated_image)
    output_file = f"{sPROCESSED_OUTPUT_DATA_PATH}/{observation_id}".strip(".fits")
    np.savez_compressed(output_file,cropped_image)
    plt.imshow(stretch(cropped_image),cmap="gray")
    plt.savefig(f"{output_file}.png")
def getImages(observations):
    num = len(observations) if args.num==-1 else args.num
    for i in tqdm(range(num)):
        ST_1 = time.time()
        hdul=download(observations[i])
        print("Time taken download: ", time.time() - ST_1, "seconds")
        if hdul:
            ST_2 = time.time()
            preprocess(hdul,observations[i])
            print("Time taken preprocess: ", time.time() - ST_2, "seconds")

def getObservations():
    with open(OBSERVATIONS_PATH) as f:
        observations = f.read().splitlines()
    return observations

def main():
    observations=getObservations()
    getImages(observations)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Query the Hubble Legacy Archive')
    parser.add_argument('--calibration_level', type=str, default='PRODUCT',
                        help='Calibration level of the data')
    parser.add_argument('--data_product_type', type=str, default='image',
                        help='Type of data product')
    parser.add_argument('--intent', type=str, default='SCIENCE',
                        help='Observation intent')
    parser.add_argument('--query_id', type=str, default='tmp',
                        help='Dataset/query identifier')
    parser.add_argument('--num', type=int, default=-1,
                        help='Num Images to retrieve')
    parser.add_argument('--prep', action='store_true',
                        help='Do Preprocessing or Not')
    parser.add_argument('--crop_margin', type=float, default=0.05,
                        help='Percitle Crop of Width and Height of a single image, to remove all background')
   
    args = parser.parse_args()    
    hubbler = ESAHubble()
    sRAW_DATA_PATH = os.path.join(RAW_DATA_PATH,f"{args.query_id}")
    OBSERVATIONS_PATH = os.path.join(QUERY_PATH,f"{args.query_id}/observation_id.txt")
    if not os.path.isdir(sRAW_DATA_PATH): os.mkdir(sRAW_DATA_PATH)
    sPROCESSED_OUTPUT_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,f"{args.query_id}")
    stretch = LogStretch()
    if not os.path.isdir(sPROCESSED_OUTPUT_DATA_PATH): os.mkdir(sPROCESSED_OUTPUT_DATA_PATH)

    main()
