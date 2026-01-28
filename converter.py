import tensorflow as tf

model_path = 'model_kualitas_kopi.h5'
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_model_path = 'model_kopi.tflite'

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("\n [INFO] Model has convert to tflite")