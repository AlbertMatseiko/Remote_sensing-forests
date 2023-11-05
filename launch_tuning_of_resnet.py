from tune_script import tune_res_net

# Global variables, do not change (HyperHyperparameters)
path_to_h5 = './DATA/h5_files/LC08_L2SP_02_T1_256.h5'
WIDTH = 256
HEIGHT = 256
CHANNELS = 7
CLASSES = 10
MAX_SHIFT = 1  # пока не используем
BATCH_SIZE = 4
DEPTH = 2
CONTRAST_FACTOR = 0.1
model_name = f"ResNet_to_tune_DEPTH{DEPTH}"

dict_of_args = {"DEPTH": DEPTH,
                "CHANNELS": CHANNELS,
                "CLASSES": CLASSES,
                "WIDTH": WIDTH,
                "HEIGHT": HEIGHT,
                "BATCH_SIZE": BATCH_SIZE,
                "CONTRAST_FACTOR": CONTRAST_FACTOR,
                "model_name": model_name,
                "path_to_h5": path_to_h5}

trigger1 = input("Do you want to tune lr? Print only y or n")
if trigger1 == 'y':
    _ = tune_res_net(regime="lr", **dict_of_args)
    print("Report of lr tune is done!")
else:
    print("Since answer is not 'y', no lr tuning is provided.")

trigger2 = input("Do you want to tune hp? Print only y or n")
if trigger2 == 'y':
    _ = tune_res_net(regime="lr", **dict_of_args)
    print("Report of hp tune is done!")
else:
    print("Since answer is not 'y', no hp tuning is provided.")

if trigger1 != 'y' and trigger2 != 'y':
    print("So what the F*CK do you want here?!?! I'm DONE!")
