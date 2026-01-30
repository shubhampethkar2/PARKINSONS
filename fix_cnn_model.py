import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# Load original model WITHOUT compiling
old_model = load_model("models/cnn_model.h5", compile=False)

# Create new input layer WITHOUT batch_shape
new_input = Input(shape=(224, 224, 3))

# Rebuild model graph
new_output = old_model(new_input)
new_model = Model(inputs=new_input, outputs=new_output)

# Save in compatible format
new_model.save("models/cnn_model_fixed.h5")

print("CNN model re-saved successfully as cnn_model_fixed.h5")
