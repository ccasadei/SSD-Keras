import os
from math import ceil

from keras.utils import plot_model

from config import Config
from model.callbacks import get_callbacks
from model.generator import get_generators
from model.loss import getLoss
from model.optimizer import getOptimizer
from model.ssd300 import ssd_300
from model.ssd512 import ssd_512

# leggo la configurazione
config = Config('configSSD.json')

# creazione del modello
wpath = None
if config.type == '300':
    wpath = config.base_weights_path300
    model, bodyLayers, predictor_sizes = ssd_300(n_classes=len(config.classes) - 1,  # NOTA: qui tolgo la classe di background nel conteggio
                                                 scales=config.scales,
                                                 aspect_ratios_per_layer=config.aspect_ratios_per_layer,
                                                 steps=config.steps,
                                                 offsets=config.offsets)

else:
    wpath = config.base_weights_path512
    model, bodyLayers, predictor_sizes = ssd_512(n_classes=len(config.classes) - 1,  # NOTA: qui tolgo la classe di background nel conteggio
                                                 scales=config.scales,
                                                 aspect_ratios_per_layer=config.aspect_ratios_per_layer,
                                                 steps=config.steps,
                                                 offsets=config.offsets)

model.summary()

# verifico se esistono dei pesi pre-training
if os.path.isfile(config.pretrained_weights_path):
    model.load_weights(config.pretrained_weights_path, by_name=True, skip_mismatch=True)
    print("Caricati pesi PRETRAINED")

    # altrimenti carico i pesi di base (escludendo i layer successivi a quello indicato, compreso)
elif os.path.isfile(wpath):
    model.load_weights(wpath, by_name=True, skip_mismatch=True)
    print("Caricati pesi BASE")
else:
    print("Senza pesi")

# eseguo il freeze dei layer più profondi
if config.do_freeze_layers:
    if config.type == '300':
        freeze_pops = config.freeze_pops300
    else:
        freeze_pops = config.freeze_pops512

    for l in bodyLayers[:len(bodyLayers) - freeze_pops]:
        model.get_layer(l).trainable = False
    print("")
    print("Eseguito freeze di " + str(len(bodyLayers) - freeze_pops) + " layers")
    print("Nuovo summary dopo FREEZE")
    print("")
    model.summary()

# compilo il model con lossfunction e ottimizzatore
model.compile(loss=getLoss(), optimizer=getOptimizer(config.base_lr), metrics=['accuracy'])

if config.model_image:
    plot_model(model, to_file='model_image.jpg')

train_generator, val_generator, n_train_samples, n_val_samples = get_generators(config, model, predictor_sizes)

callbacks = get_callbacks(config)

# eseguo il training
model.fit_generator(generator=train_generator,
                    steps_per_epoch=ceil(n_train_samples / config.batch_size),
                    epochs=config.epochs,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=ceil(n_val_samples / config.batch_size))

# salvo i pesi
model.save_weights(config.trained_weights_path)
