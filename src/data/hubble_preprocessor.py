import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from astropy.io import fits
from pathing import PROCESSED_DATA_PATH,RAW_DATA_PATH
from astroquery.esa.hubble import ESAHubble
from astropy.visualization import LogStretch
from matplotlib import pyplot as plt

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




def main():
    for filename in tqdm(os.listdir(INPUT_DATA_PATH)):
        hdul = fits.open(os.path.join(INPUT_DATA_PATH,filename))
        i = 1 if len(hdul) > 1 else 0
        image = hdul[i].data
        try:
            exp_time = hdul[0].header["EXPTIME"]
            rotated_image = findRotateImage(image)
            cropped_image = crop(rotated_image)
            output_file = f"{OUTPUT_DATA_PATH}/{filename}_{exp_time}".replace(".fits","")
            np.savez_compressed(output_file,cropped_image)
            plt.imshow(stretch(cropped_image),cmap="gray")
            plt.savefig(f"{output_file}.png")
        except Exception as e:
            print(f"Error: {e}")
            print("No Exposure-Time found! Not saved!")
        
    
    pass
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Query the Hubble Legacy Archive')
    parser.add_argument('--crop_margin', type=float, default=0.05,
                        help='Percitle Crop of Width and Height of a single image, to remove all background')
    parser.add_argument('--query_id', type=str, default="hubble_base",
                        help='Dataset/Query identifier')

    args = parser.parse_args()    
    hubbler = ESAHubble()
    INPUT_DATA_PATH = os.path.join(RAW_DATA_PATH,f"{args.query_id}")
    OUTPUT_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,f"{args.query_id}")
    stretch = LogStretch()
    if not os.path.isdir(OUTPUT_DATA_PATH): os.mkdir(OUTPUT_DATA_PATH)
    

    main()