import json


class Config:

    def __init__(self, config_path):
        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())

        self.batch_size = config['train']['batch_size']
        self.epochs = config['train']['epochs']
        self.base_lr = config['train']['base_lr']
        self.patience = config['train']['patience']
        self.do_freeze_layers = config['train']['do_freeze_layers']
        self.train_val_split = config['train']['train_val_split']
        self.freeze_layer_stop_name_300 = config['train']['freeze_layer_stop_name_300']
        self.freeze_layer_stop_name_512 = config['train']['freeze_layer_stop_name_512']
        self.augmentation = config['train']['augmentation']

        self.pretrained_weights_path = config['path']['pretrained_weights']
        self.base_weights_path_300 = config['path']['base_weights_300']
        self.base_weights_path_512 = config['path']['base_weights_512']
        self.trained_weights_path = config['path']['trained_weights']
        self.chkpnt_weights_path = config['path']['chkpnt_weights']
        self.images_path = config['path']['images']
        self.annotations_path = config['path']['annotations']
        self.test_images_path = config['path']['test_images']
        self.test_result_path = config['path']['test_result']
        self.log_path = config['path']['log']

        self.model_image = config['model']['model_image']
        self.type = config['model']['type']
        self.classes = config['model']['classes']

        if self.type == "300":
            self.scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
            self.aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5]]
            self.steps = [8, 16, 32, 64, 100, 300]
            self.offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        elif self.type == "512":
            self.scales = [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
            self.aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5]]
            self.steps = [8, 16, 32, 64, 128, 256, 512]
            self.offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        else:
            print("Tipo di architettura non consciuta '" + self.type + '"')
            exit()

        print("Architettura " + self.type)
