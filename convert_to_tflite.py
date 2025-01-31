import tensorflow as tf

saved_model_path = 'pokemon_classifier.h5'
converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(saved_model_path))
tflite_model = converter.convert()

with open('pokemon_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as pokemon_classifier.tflite")
