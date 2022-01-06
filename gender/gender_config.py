# PATHS
GENDER_DATA = 'D:/data/face_detection/utk/gender'
LOG_PATH = "./logs"
MODEL_PATH = './models/val_loss_{val_loss:.3f}.hdf5'

# Image Variables
IMG_SIZE = (200,200)
COLOR_MODE = "rgb"
IMG_SHAPE = IMG_SIZE + (3,) if COLOR_MODE == "rgb" else IMG_SIZE + (1,)

# Training variables
VAL_SPLIT = 0.15
BATCH_SIZE = 64
LEARNING_RATE = 0.02
EPSILON = 0.05
DECAY = 0.9
