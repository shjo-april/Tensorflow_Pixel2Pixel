
import cv2
import numpy as np

from Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def Get_Domain_Data(image_path, option):
    if option == 'color':
        dst_image = cv2.imread(image_path)
        dst_image = cv2.resize(dst_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        src_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    
    elif option == 'facade':
        dst_image = cv2.imread(image_path)
        dst_image = cv2.resize(dst_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        src_image = cv2.imread(image_path.replace('/image', '/png').replace('.jpg', '.png'))
        src_image = cv2.resize(src_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    return src_image, dst_image

def Save(fake_images, save_path):
    save_image = np.zeros((IMAGE_HEIGHT * SAVE_HEIGHT, IMAGE_WIDTH * SAVE_WIDTH, OUTPUT_IMAGE_CHANNEL), dtype = np.uint8)
    
    for y in range(SAVE_HEIGHT):
        for x in range(SAVE_WIDTH):
            fake_image = (fake_images[y * SAVE_WIDTH + x] + 1) * 127.5
            save_image[y * IMAGE_HEIGHT : (y + 1) * IMAGE_HEIGHT, x * IMAGE_WIDTH : (x + 1) * IMAGE_WIDTH] = fake_image.astype(np.uint8)

    cv2.imwrite(save_path, save_image)
