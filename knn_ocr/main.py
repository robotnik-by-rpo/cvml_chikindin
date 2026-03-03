import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread

def phrase(res, map_char,idx,spaces):
    str_res = idx+": "  
    for i,r in enumerate(res):
        str_res += map_char[float(r[0])]
        if i+1 in spaces:
            str_res += " "
    return str_res


def make_train(path):
    cls_map = {}
    train = []
    responses = []
    ncls = 0
    for cls in sorted(path.glob("*")):
        ncls += 1
        if len(cls.name)== 2:
            cls_map[float(ncls)] = cls.name[1]
        else:
            cls_map[float(ncls)] = cls.name[0]
        
        for p in cls.glob("*.png"):
            train.append(extractor(imread(p)))
            responses.append(ncls)
    train = np.array(train,dtype = "f4").reshape(-1,9)
    responses = np.array(responses, dtype = "f4").reshape(-1,1)
    return train, responses,cls_map

def extractor(image):
    if image.ndim == 2:
        binary = image
    else:

        gray = np.mean(image,axis = 2).astype('u1')
        binary = gray > 0
    lb = label(binary)
    props = max(regionprops(lb), key=lambda r: r.area)
    return np.hstack([props.moments_hu, [props.eccentricity,props.solidity]], dtype="f4")

data = Path("./task")

#Итоговые результаты полученных слов
images_phrase = []

for im in sorted(data.glob("*.png")):
    print(im.name)
    spaces_index = []
    last_col = None

    image = imread(im)

    #Обучающий датасет
    train, responses,res_map = make_train(data/"train")
    knn = cv2.ml.KNearest.create()
    knn.train(train,cv2.ml.ROW_SAMPLE, responses)

    #Тестовый датасет
    gray = image.mean(2)
    binary = gray > 0
    find = []
    lb = label(binary)
    char_idx = 0
    props = sorted(regionprops(lb), key = lambda c: c.centroid[1])
    for idx, prop in enumerate(props):
        if prop.area < 300: continue
        find.append(extractor(prop.image))

        _, min_col, _, max_col = prop.bbox
        if last_col is not None:
            if min_col - last_col > 20:
                spaces_index.append(char_idx)
                
        last_col = max_col
        char_idx += 1
    find = np.array(find, dtype = "f4").reshape(-1,9)
    ret, results, neightbours, dist = knn.findNearest(find,5)

    images_phrase.append(phrase(results,res_map,im.name,spaces_index))

print("\n",*images_phrase,sep="\n")
