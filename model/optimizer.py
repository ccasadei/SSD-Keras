from keras.optimizers import Adam, SGD


def getOptimizer(base_lr):
    # return SGD(lr=base_lr, momentum=0.9)
    return Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
