import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import shutil
import PIL

from fastai import *
from fastai.vision import *

def get_counts(path): return (path.stem, pd.read_csv(path).shape[0])
with ThreadPoolExecutor(32) as e: counts = list(e.map(get_counts, Path('data/train').iterdir()))

sorted_counts = sorted(counts, key=lambda tup: tup[1])

print(np.sum([tup[1] for tup in counts]))
print(len(counts))

BASE_SIZE = 256
def draw_cv2(raw_strokes, size=256, lw=5, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    img = cv2.copyMakeBorder(img,4,4,4,4,cv2.BORDER_CONSTANT)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return img
        
sz = 128 # size
r = 0.5 # portion of images to keep, we want only 1% of total train data
shutil.rmtree(f'data/train-{sz}', ignore_errors=True)
os.makedirs(f'data/train-{sz}')

def save_ims_from_df(path):
    df = pd.read_csv(path)
    selected = df[df.recognized==True].sample(int(r * df.shape[0]))
    print("in")
    for row in selected.iterrows():
        idx, drawing, label = row[0], eval(row[1].drawing), '_'.join(row[1].word.split())
        ary = draw_cv2(drawing, size=128)
        rgb_ary = np.repeat(ary[:,:,None], 3, -1)
        PIL.Image.fromarray(rgb_ary).save(f'data/train-{sz}/{label}_{idx}.png')
    print("out")

with ThreadPoolExecutor(32) as e: e.map(save_ims_from_df, Path('data/train').iterdir())

