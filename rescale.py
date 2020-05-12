# load the image and rescale
import os
import cv2

data = "data-original"

def rescale(path,file,tumour):
    original_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    resized_image = cv2.resize(original_image, (128, 128), 
                            interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("{0}/{1}/resized/{2}".format(data,tumour,file),resized_image)


for root, dirs, files in os.walk('{0}/no'.format(data)):
    for file in files:
        p=os.path.join(root,file)
        print("rescaling ",p)
        rescale(p, file, "no")

for root, dirs, files in os.walk('{0}/yes'.format(data)):
    for file in files:
        p=os.path.join(root,file)
        print("rescaling ",p)
        rescale(p, file, "yes")

