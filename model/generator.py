import os
import random

from model.batch_generator import BatchGenerator
from model.box_encode_decode_utils import SSDBoxEncoder


def get_generators(config, model, predictor_sizes):
    # istanzio i generatori di batch per train e validate
    train_dataset = BatchGenerator()
    val_dataset = BatchGenerator()

    # ottengo l'elenco di tutte le annotations
    annotation_files = [os.path.splitext(f)[0] for f in os.listdir(config.annotations_path) if os.path.isfile(os.path.join(config.annotations_path, f))]
    # mescola l'ordine delle righe
    random.shuffle(annotation_files)
    max_id = int(config.train_val_split * len(annotation_files))
    train_ids = annotation_files[:max_id]
    if config.train_val_split < 1.:
        val_ids = annotation_files[max_id:]
    else:
        val_ids = None

    size = int(config.type)

    train_dataset.parse_xml(images_dirs=[config.images_path],
                            image_set_filenames=train_ids,
                            annotations_dirs=[config.annotations_path],
                            classes=config.classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[config.images_path],
                          annotations_dirs=[config.annotations_path],
                          image_set_filenames=val_ids,
                          classes=config.classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)

    ssd_box_encoder = SSDBoxEncoder(img_height=size,
                                    img_width=size,
                                    n_classes=len(config.classes) - 1,
                                    predictor_sizes=predictor_sizes,
                                    scales=config.scales,
                                    aspect_ratios_per_layer=config.aspect_ratios_per_layer,
                                    steps=config.steps,
                                    offsets=config.offsets)

    # impostao le opzioni di estrazione/manipolazione per le immagini da generare per train e validate
    train_generator = train_dataset.generate(batch_size=config.batch_size,
                                             shuffle=True,
                                             train=True,
                                             ssd_box_encoder=ssd_box_encoder,
                                             augmentation=config.augmentation,
                                             equalize=True,
                                             brightness=(0.5, 2, 0.5),
                                             flip=0.5,
                                             translate=((20, 20), (20, 20), 0.5),
                                             scale=(0.9, 1.1, 0.5),
                                             max_crop_and_resize=(size, size, 1, 3),
                                             random_crop=(size, size, 1, 3, 0.5),
                                             crop=False,
                                             resize=False,
                                             gray=False,
                                             limit_boxes=True,
                                             include_thresh=0.4)

    val_generator = val_dataset.generate(batch_size=config.batch_size,
                                         shuffle=True,
                                         train=False,
                                         ssd_box_encoder=ssd_box_encoder,
                                         augmentation=False,
                                         max_crop_and_resize=(size, size, 1, 3),
                                         random_crop=(size, size, 1, 3, 0.5))

    n_train_samples = train_dataset.get_n_samples()
    n_val_samples = val_dataset.get_n_samples()

    return train_generator, val_generator, n_train_samples, n_val_samples
