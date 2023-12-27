import os
from mira.detectors import MTCNN
from keras_facenet import FaceNet
import cv2
import numpy as np
from scipy.spatial import distance
import time

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

o_path = os.getcwd()
os.chdir(os.getcwd() + '\\train')
path = os.getcwd()
dic = os.listdir()
detector = MTCNN()
embedder = FaceNet()

aemb = []
aaaa = []

for i in range(len(dic)):
    os.chdir(path + '\\' + dic[i])
    dic_2 = os.listdir()
    emb = []

    for j in range(len(dic_2)):
        print(dic_2[j])
        img = cv2.imread(dic_2[j])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect(img)
        embeddings = embedder.embeddings([
            face.selection.extract(img) for face in faces
        ])
        serialize1 = np.concatenate(embeddings)
        embs = l2_normalize(serialize1)
        emb.append(embs)
    print(dic[i])
    print(len(emb))
    ems = emb[0]
    for j in range(1, len(emb)):
        ems += emb[j]

    ems /= len(emb)
    aemb.append(ems)


os.chdir(o_path + '\\test')

dic_2 = os.listdir()

for i in range(len(dic_2)):
    img = cv2.imread(dic_2[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(dic_2[i])
    faces = detector.detect(img)
    #print(len(faces))
    embeddings = embedder.embeddings([
        face.selection.extract(img) for face in faces
    ])
    serialize1 = np.concatenate(embeddings)
    embs = l2_normalize(serialize1)
    aaaa.append(embs)

din = []

t1 = time.time()
for i in range(len(aaaa)):
    o = []
    for j in range(len(aemb)):
        p = []
        p.append(distance.euclidean(aaaa[i], aemb[j]))
        p.append(dic[j])
        o.append(p)
    din.append(o)

for i in range(len(din)):
    din[i].sort(key=lambda s: s[0])

t2 = time.time()

print('cost ' + str(t2 - t1) + 's')

os.chdir(o_path)
f = open('result.txt', 'w')
count = 0
for i in range(len(dic_2)):
    check = False
    str1 = dic_2[i]
    str2 = din[i][0][1]
    if str1.find(str2) != -1:
        check = True
    if check is True:
        count += 1
    f.write(dic_2[i] + ", " + str(din[i][0]) + ' ' + str(check) + '\n')
    print(dic_2[i], din[i][0], check)

f.write('test data = ' + str(len(dic_2)) + ' true = ' + str(count) + '\n')
f.write('accuracy is ' + str(count / len(dic_2)) + '\n')
print('test data = ' + str(len(dic_2)) + ' true = ' + str(count))
print('accuracy is ' + str(count / len(dic_2)))

f.close()

v = input('input any key to end')

