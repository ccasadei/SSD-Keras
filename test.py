import matplotlib as mpl

mpl.use('Agg')

import numpy as np
from keras.preprocessing import image
from scipy.misc import imread

from matplotlib import pyplot as plt

from config import Config
from model.optimizer import getOptimizer
from model.ssd300 import ssd_300
from model.ssd512 import ssd_512
from model.loss import getLoss
from model.box_encode_decode_utils import decode_y
import os

# leggo la configurazione
config = Config('configSSD.json')

# se non ci sono pesi specifici, uso i pesi base e le classi base (COCO)
wname = 'BASE'
if config.type == '300':
    wpath = config.base_weights_path_300
elif config.type == '512':
    wpath = config.base_weights_path_512
else:
    print("Tipo di architettura non consciuta '" + config.type + '"')
    exit()

classes = ['background',
           'person',
           'bicycle',
           'car',
           'motorcycle',
           'airplane',
           'bus',
           'train',
           'truck',
           'boat',
           'traffic light',
           'fire hydrant',
           'stop sign',
           'parking meter',
           'bench',
           'bird',
           'cat',
           'dog',
           'horse',
           'sheep',
           'cow',
           'elephant',
           'bear',
           'zebra',
           'giraffe',
           'backpack',
           'umbrella',
           'handbag',
           'tie',
           'suitcase',
           'frisbee',
           'skis',
           'snowboard',
           'sports ball',
           'kite',
           'baseball bat',
           'baseball glove',
           'skateboard',
           'surfboard',
           'tennis racket',
           'bottle',
           'wine glass',
           'cup',
           'fork',
           'knife',
           'spoon',
           'bowl',
           'banana',
           'apple',
           'sandwich',
           'orange',
           'broccoli',
           'carrot',
           'hot dog',
           'pizza',
           'donut',
           'cake',
           'chair',
           'couch',
           'potted plant',
           'bed',
           'dining table',
           'toilet',
           'tv',
           'laptop',
           'mouse',
           'remote',
           'keyboard',
           'cell phone',
           'microwave',
           'oven',
           'toaster',
           'sink',
           'refrigerator',
           'book',
           'clock',
           'vase',
           'scissors',
           'teddy bear',
           'hair drier',
           'toothbrush']

# se invece ci sono pesi specifici, uso questi pesi e le classi per cui sono stati trovati
if os.path.isfile(config.trained_weights_path):
    wname = "DEFINITIVI"
    wpath = config.trained_weights_path
    classes = config.classes
elif os.path.isfile(config.pretrained_weights_path):
    wname = 'PRETRAINED'
    wpath = config.pretrained_weights_path
    classes = config.classes

# creo il modello
if config.type == '300':
    model, _, _ = ssd_300(n_classes=len(classes)-1)
else:
    model, _, _ = ssd_512(n_classes=len(classes)-1)

# carico i pesi
model.load_weights(wpath, by_name=True, skip_mismatch=True)
print("Caricati pesi " + wname)

# compilo il model con lossfunction e ottimizzatore
model.compile(loss=getLoss(), optimizer=getOptimizer(config.base_lr))

img_height = model.input_shape[1]
img_width = model.input_shape[2]

# carico le immagini originali e quelle ridimensionate in due array
# ne prendo una alla volta per minimizzare la memoria GPU necessaria
for imgf in os.listdir(config.test_images_path):
    imgfp = os.path.join(config.test_images_path, imgf)
    if os.path.isfile(imgfp):
        orig_image = imread(imgfp)
        input_image = []
        img = image.load_img(imgfp, target_size=(img_height, img_width))
        input_image.append(image.img_to_array(img))
        input_image = np.array(input_image)

        # eseguo la perdizione sulle immagini ridimensionate
        y_pred = model.predict(input_image)

        # decodifico il risultato restituendo solo i positivi
        y_pred_decoded = decode_y(y_pred,
                                  confidence_thresh=0.1,
                                  iou_threshold=0.45,
                                  top_k=200,
                                  normalize_coords=True,
                                  img_height=img_height,
                                  img_width=img_width)

        # prepara i colori
        colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()

        # scrivo i box
        plt.imshow(orig_image)

        current_axis = plt.gca()

        if len(y_pred_decoded) > 0:
            for box in y_pred_decoded[0]:
                # trasformo le coordinate normalizzate in coordinate assolute
                xmin = box[-4] * orig_image.shape[1] / img_width
                ymin = box[-3] * orig_image.shape[0] / img_height
                xmax = box[-2] * orig_image.shape[1] / img_width
                ymax = box[-1] * orig_image.shape[0] / img_height
                color = colors[int(box[0]) % len(colors)]
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
        plt.savefig(os.path.join(config.test_result_path, imgf))
        plt.close()
        print("Elaborata immagine '" + imgf + "'")
