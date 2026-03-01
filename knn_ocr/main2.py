import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
# from skimage.morphology import binary_dilation, disk,binary_opening, binary_closing, binary_erosion
from collections import defaultdict

def make_train(path):
    cls_map = {}
    train = []
    responses = []
    ncls = 0
    for cls in sorted(path.glob("*")):
        ncls += 1
        
        # if cls.name[0] =="s" and len(cls.name) == 2:
        #     cls.name = cls.name[1]
        print(cls.name,ncls)
        for p in cls.glob("*.png"):
            # print(p)
            train.append(extractor(imread(p)))
            responses.append(ncls)
    train = np.array(train,dtype = "f4").reshape(-1,5)
    responses = np.array(responses, dtype = "f4").reshape(-1,1)
    return train, responses

def extractor(image):
    if image.ndim == 2:
        binary = image
    else:

        gray = np.mean(image,axis = 2).astype('u1')
        binary = gray <255
    lb = label(binary)
    props = regionprops(lb)[0]

    cy,cx = props.centroid_local
    shape = props.image.shape

    return np.array([props.eccentricity,
                     props.area / np.pi**0.5,
                     props.area / props.perimeter*1.5,
                     cy / shape[0],
                     cx / shape[1]
                     ],dtype="f4")

data = Path("./task")
#Итоговые результаты полученных слов
images_phrase = []

for im in sorted(data.glob("*.png")):
    image = imread(im)

    #Обучающий датасет
    train, responses = make_train(data/"train")
    knn = cv2.ml.KNearest.create()
    knn.train(train,cv2.ml.ROW_SAMPLE, responses)

    #Тестовый дата сет
    gray = image.mean(2)
    binary = gray > 0
    find = []
    lb = label(binary)
    props = regionprops(lb)
    for prop in props:
        find.append(extractor(prop.image))
    find = np.array(find, dtype = "f4").reshape(-1,5)
    ret, results, neightbours, dist = knn.findNearest(find,5) 

print(images_phrase)