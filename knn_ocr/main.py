import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
# from skimage.morphology import binary_dilation, disk,binary_opening, binary_closing, binary_erosion
from collections import defaultdict

"""подправить кодирование строчных букв из файлов для корректного названия классов"""

def preper_word(words):
    res = ""
    for w in words:
        if len(w)==2 and w[0] == 's':
            res += w[1]
        else:
            res += w[0]
    return res

def segment_text_vertical_projection(image):

    binary = image.copy()
    # Вертикальная проекция (сумма пикселей по вертикали)
    vertical_projection = np.sum(binary, axis=0) // 255
    
    # Поиск границ символов
    chars = []
    in_char = False
    start = 0
    
    for i, value in enumerate(vertical_projection):
        if value > 0 and not in_char:
            in_char = True
            start = i
        elif value == 0 and in_char:
            in_char = False
            end = i
            # Проверка ширины символа (отсеиваем шум)
            if end - start > 5:  # минимальная ширина символа
                chars.append((start, end))
    
    # Анализ пробелов между символами
    spaces = []
    for i in range(len(chars) - 1):
        gap = chars[i+1][0] - chars[i][1]
        if gap > 20:  # порог для определения пробела между словами
            spaces.append(i)
    
    return chars, spaces, binary

def make_train(path):
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
    train = np.array(train,dtype = "f4").reshape(-1,2)
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
    gray = gray > 0
    find = []
    chars, _, _ = segment_text_vertical_projection(image)
    for c in chars:
        find.append(extractor(c))
    find = np.array(find, dtype = "f4").reshape(-1,2)
    ret, results, neightbours, dist = knn.findNearest(find,5) 
    # images_phrase.append("".join(results))
    images_phrase.append(preper_word(results))
    plt.imshow(gray)
    plt.show()
print(images_phrase)