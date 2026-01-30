import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Rebuild the SAME architecture used during training
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None   # IMPORTANT
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

new_model = Model(inputs=base_model.input, outputs=output)

# Load weights from old model (THIS WORKS EVEN IF MODEL IS BROKEN)
new_model.load_weights("models/cnn_model.h5")

# Save clean model
new_model.save("models/cnn_model_clean.h5")

print("CNN model rebuilt and saved as cnn_model_clean.h5")
