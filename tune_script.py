import tensorflow as tf
import keras_tuner as kt
import TrainingNN.BuildNN as BuildNN
import numpy as np
import h5py as h5
from TrainingNN.DataLoad import make_train_dataset


# Model to tune HP in keras-tuner
class ResNetHyperModel(kt.HyperModel):
    # initialization with globals
    def __init__(self, lr_opt=True, lr=0.005, DEPTH=2,
                 CHANNELS=7, CLASSES=10, WIDTH=256, HEIGHT=256, BATCH_SIZE=8,
                 CONTRAST_FACTOR=0.1, default_filters=None, default_kernels=None):
        super().__init__()
        # set default hyper params to optimize lr first
        if default_kernels is None:
            default_kernels = [3] * 5
        if default_filters is None:
            default_filters = [32] * 5
        default_hps = {"filters": default_filters,
                       "conv_kernel": default_kernels,
                       "DEPTH": DEPTH,
                       "CHANNELS": CHANNELS,
                       "CLASSES": CLASSES,
                       "WIDTH": WIDTH,
                       "HEIGHT": HEIGHT,
                       "BATCH_SIZE": BATCH_SIZE,
                       "CONTRAST_FACTOR": CONTRAST_FACTOR}
        self.default_hps = default_hps
        self.hps_dict = None
        self.lr_opt = lr_opt
        self.lr = lr
        self.DEPTH = DEPTH
        self.BATCH_SIZE = BATCH_SIZE
        self.CHANNELS = CHANNELS
        self.CLASSES = CLASSES
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CONTRAST_FACTOR = CONTRAST_FACTOR

    def customise_HP(self, hp):
        # if lr optimize is True, it is optimized
        if self.lr_opt:
            self.lr = hp.Float('lr_i', 1e-5, 0.1, step=10, sampling="log")
            self.decay_rate = 0.9
            self.hps_dict = self.default_hps
        # if lr_opt False, optimize hps with given lr
        else:
            self.hps_dict = {"filters": [hp.Choice(f"filters_{i}", [16, 64, 128]) for i in range(self.DEPTH)],
                             "conv_kernel": [hp.Choice(f"conv_kernel_{i}", [3, 5]) for i in range(self.DEPTH)],
                             "DEPTH": self.DEPTH,
                             "CHANNELS": self.CHANNELS,
                             "CLASSES": self.CLASSES,
                             "WIDTH": self.WIDTH,
                             "HEIGHT": self.HEIGHT,
                             "BATCH_SIZE": self.BATCH_SIZE,
                             "CONTRAST_FACTOR": self.CONTRAST_FACTOR}

    # builds the model to tune
    def build(self, hp):
        self.customise_HP(hp)
        model, model_name = BuildNN.make_resnet_model(**self.hps_dict)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                                    amsgrad=False, name='Adam')
        model.compile(optimizer=optimizer)
        return model

    # fit function
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=self.BATCH_SIZE, verbose=False, **kwargs)


# procedure that optimizes the lr or hps, depending on "regime"
def tune_res_net(hm_model=ResNetHyperModel, path_to_h5=None, model_name="Not_a_name", regime="lr",
                 BATCH_SIZE=8, WIDTH=256, HEIGHT=256, DEPTH=2, CHANNELS=7, CLASSES=10, CONTRAST_FACTOR=0.1,
                 cutting=10, num_of_epochs=1, max_lr_trials=10, max_hp_trials=50,
                 project_name_lr="tune_lr", project_name_hp="tune_hp"):
    assert path_to_h5 is not None
    dict_of_args = {"DEPTH": DEPTH,
                    "CHANNELS": CHANNELS,
                    "CLASSES": CLASSES,
                    "WIDTH": WIDTH,
                    "HEIGHT": HEIGHT,
                    "BATCH_SIZE": BATCH_SIZE,
                    "CONTRAST_FACTOR": CONTRAST_FACTOR}

    with h5.File(path_to_h5, 'r') as f:
        total_num = f['all/data_norm'].shape[0]
        steps_per_epoch = (total_num // BATCH_SIZE) // cutting

    train_dataset = make_train_dataset(path_to_h5, BATCH_SIZE, WIDTH, HEIGHT, CHANNELS)
    path_to_report = './models/' + model_name + '/tuning/'
    # tune my learning rate with loss parameters
    if regime == "lr":
        tuner = kt.RandomSearch(
            hm_model(lr_opt=True, **dict_of_args),
            objective=kt.Objective("loss", direction="min"),
            max_trials=max_lr_trials,
            overwrite=True,
            directory=path_to_report,
            project_name=project_name_lr,
            max_retries_per_trial=0,
        )
        _ = tuner.search(train_dataset, epochs=num_of_epochs, steps_per_epoch=steps_per_epoch)

        report_file = open(path_to_report + "info_tune_lr.txt", "w")
        best_from_lr_tune = tuner.get_best_hyperparameters()[0].values
        best_lr_metric = tuner.oracle.get_best_trials(1)[0].score
        report_file.write(f"Best lr = {best_from_lr_tune} with metric = {best_lr_metric}. \n")
        report_file.close()

        return _
    else:
        try:
            tuner_lr = kt.RandomSearch(
                hm_model(lr_opt=True, **dict_of_args),
                objective=kt.Objective("loss", direction="min"),
                overwrite=False,
                directory=path_to_report,
                project_name=project_name_lr
            )
            best_from_lr_tune = tuner_lr.get_best_hyperparameters()[0].values
            best_lr = np.round(best_from_lr_tune['lr_i'], 5)
            print("LOADED BEST LR=", best_lr)
        except:
            print("Failed to load best lr, set 0.1 as default.")
            best_lr = 0.1

        tuner_hp = kt.RandomSearch(
            hm_model(lr_opt=False, lr=best_lr, **dict_of_args),
            objective=kt.Objective("loss", direction="min"),
            max_trials=max_hp_trials,
            overwrite=True,
            directory=path_to_report,
            project_name=project_name_hp
        )
        _ = tuner_hp.search(train_dataset, epochs=num_of_epochs, steps_per_epoch=steps_per_epoch)

        report_file = open(path_to_report + "info_tune_hp.txt", "w")
        best_from_hp_tune = tuner_hp.get_best_hyperparameters()[0].values
        best_hp_metric = tuner_hp.oracle.get_best_trials(1)[0].score
        report_file.write(f"Best hp = {best_from_hp_tune} with metric = {best_hp_metric}. \n")
        report_file.close()

        return _
