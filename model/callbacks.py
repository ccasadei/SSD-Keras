from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau


def getCallbacks(config):
    # definisco uno scheduler per il learning rate
    def lr_schedule(epoch, decay=0.9):
        # new_lr = config.base_lr * (decay ** epoch)
        if epoch < 80:
            new_lr = config.base_lr
        elif epoch < 100:
            new_lr = config.base_lr / 10.
        else:
            new_lr = config.base_lr / 100.
        print("Learning Rate: " + str(new_lr))
        return new_lr

    return [
        ModelCheckpoint(config.chkpnt_weights_path,
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='auto',
                        period=1),
        LearningRateScheduler(lr_schedule),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
        EarlyStopping(monitor='val_loss',
                      min_delta=0.0001,
                      patience=config.patience),
        TensorBoard(log_dir=config.log_path)]
