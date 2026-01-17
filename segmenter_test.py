# segmenter_test.py

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure the root project directory is in the path
sys.path.append(os.path.abspath("."))

from CNN.segmenter import preprocess_frame, decode_segmentation_mask, class_colors

def main():
    # Paths to SavedModel directory and test image
    MODEL_PATH = r"C:\Users\zmsub\OneDrive\Desktop\VBN_INTEGRATION\CNN\resnet50_unet_tf_savedmodel"
    IMAGE_PATH = r"C:\Users\zmsub\OneDrive\Desktop\GP2_VBN1\SDD\semantic_drone_dataset\training_set\images\598.jpg"

    # Debug path check
    print("MODEL_PATH exists:", os.path.isdir(MODEL_PATH))
    print("IMAGE_PATH exists:", os.path.isfile(IMAGE_PATH))

    # Check if paths exist
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # Load full model from SavedModel format
    model = tf.saved_model.load(MODEL_PATH)

    # Load and preprocess image
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess input
    input_tensor = preprocess_frame(img_rgb)

    # Run inference using the 'serving_default' signature
    infer = model.signatures["serving_default"]
    output = infer(tf.convert_to_tensor(input_tensor))

    # Extract the prediction tensor (first output key)
    prediction = list(output.values())[0].numpy()[0]  # (256, 256, NUM_CLASSES)
    label_map = np.argmax(prediction, axis=-1)        # (256, 256)

    # Resize prediction back to original resolution
    label_map_resized = cv2.resize(
        label_map.astype(np.uint8),
        (img_rgb.shape[1], img_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Decode to RGB mask
    decoded_mask = decode_segmentation_mask(label_map_resized, class_colors)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(decoded_mask)
    plt.title("Predicted Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Input shape:", img_rgb.shape)
    print("Predicted mask shape:", label_map_resized.shape)
    print("Unique predicted classes:", np.unique(label_map_resized))

if __name__ == "__main__":
    main()

