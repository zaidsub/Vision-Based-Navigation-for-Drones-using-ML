# cnn/segmenter.py

import os
import numpy as np
import cv2
#from CNN.model_arch import ResNet50_UNet, IMG_SIZE, NUM_CLASSES
import tensorflow as tf

Model = tf.keras.models.Model


# # Load the segmentation model and its weights
# def load_segmentation_model(weights_path: str, input_shape=(256, 256, 3), num_classes=5):
#     model = ResNet50_UNet(input_shape=input_shape, num_classes=num_classes)
#     model.load_weights(weights_path)
#     return model

def load_seg_model(path):
    """
    Convenience wrapper: returns (callable_model, model_type)
    model_type is 'tf' for SavedModel, 'keras' for .h5
    """
    if os.path.isdir(path) and os.path.exists(os.path.join(path, 'saved_model.pb')):
        model = tf.saved_model.load(path)
        return model, 'tf'
    else:
        model = tf.keras.models.load_model(path)
        return model, 'keras'

# Resize and normalize the input frame
def preprocess_frame(frame: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)  # (1, H, W, 3)

# Predict class label map from input frame
def segment_frame(model: Model, frame: np.ndarray) -> np.ndarray:
    input_tensor = preprocess_frame(frame)
    prediction = model.predict(input_tensor)[0]  # (H, W, num_classes)
    return np.argmax(prediction, axis=-1)  # (H, W)

# Optional: decode label map to RGB mask for visualization
class_colors = {
    0: (0, 0, 0),        # Background – Black
    1: (150, 75, 0),     # Road – Brown
    2: (0, 0, 255),      # Building – Blue
    3: (34, 139, 34),    # Terrain – Forest Green
    4: (255, 0, 0)       # Obstacle – Red
}

def decode_segmentation_mask(mask, color_map):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color
    return color_mask