import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
# from skimage.morphology import binary_dilation, disk,binary_opening, binary_closing, binary_erosion
from collections import defaultdict

def phrase(res, map_char):
    str_res = ""
    for r in res:
        str_res += map_char[float(r[0])]
    return str_res

def make_train(path):
    cls_map = {}
    train = []
    responses = []
    ncls = 0
    for cls in sorted(path.glob("*")):
        ncls += 1
        if len(cls.name)> 2:
            cls_map[float(ncls)] = cls.name[1]
        else:
            cls_map[float(ncls)] = cls.name[0]
        print(cls.name,ncls)
        for p in cls.glob("*.png"):
            train.append(extractor(imread(p)))
            responses.append(ncls)
    train = np.array(train,dtype = "f4").reshape(-1,7)
    responses = np.array(responses, dtype = "f4").reshape(-1,1)
    return train, responses,cls_map

def extractor(image):
    if image.ndim == 2:
        binary = image
    else:

        gray = np.mean(image,axis = 2).astype('u1')
        binary = gray <255
    lb = label(binary)
    props = regionprops(lb)[0]

    return props.moments_hu

data = Path("./task")
#Итоговые результаты полученных слов
images_phrase = []

for im in sorted(data.glob("*.png")):
    image = imread(im)

    #Обучающий датасет
    train, responses,res_map = make_train(data/"train")
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
    find = np.array(find, dtype = "f4").reshape(-1,7)
    ret, results, neightbours, dist = knn.findNearest(find,5)

    images_phrase.append(phrase(results, res_map))

    # print(neightbours)

print(images_phrase)