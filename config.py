TRAIN_DIR = "/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset/Train"
VAL_DIR   = "/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset/Validation"
TEST_DIR  = "/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset/Test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

LEARNING_RATE = 1e-4

MODEL_SAVE_PATH = "models/deepfake_model.h5"

# threshold mặc định (fallback)
DEFAULT_THRESHOLD = 0.4