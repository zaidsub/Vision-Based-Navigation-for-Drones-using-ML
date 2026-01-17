import tensorflow as tf

load_model = tf.keras.models.load_model
model_from_json = tf.keras.models.model_from_json

try:
    # Try loading it as a full model
    model = load_model(r"C:\Users\zmsub\OneDrive\Desktop\VBN_INTEGRATION\CNN\resnet50_unetfinal.weights.h5", compile=False)
    print("✅ Full model was saved — use load_model()")
except Exception as e:
    print("❌ Not a full model:", e)

try:
    # Try loading just weights into the architecture
    from CNN.model_arch import ResNet50_UNet
    model = ResNet50_UNet()
    model.load_weights(r"C:\Users\zmsub\OneDrive\Desktop\VBN_INTEGRATION\CNN\resnet50_unetfinal.weights.h5" )
    print("✅ Weights loaded successfully")
except Exception as e:
    print("❌ Failed to load weights:", e)