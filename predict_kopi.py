import tensorflow as tf
import numpy as np 
from tensorflow.keras.preprocessing import image
import sys

try:
    model = tf.keras.models.load_model('model_kualitas_kopi.h5')
    print("[INFO] Model berhasil dimuat")
except Exception as e:
    print(f"[INFO] Gagal memuat model {e}")
    sys.exit(1)

class_labels = ['biji_bagus', 'biji_defek']

def prediksi_kualitas(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception as e:
        print(f"[ERROR] gagal memuat gambar: {img_path}")
        return
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = class_labels[1]
    else:
        label = class_labels[0]
    
    print(f"-- Hasil Prediksi --")
    print(f"File            : {img_path}")
    print(f"Nilai Mentah    : {prediction: .4f}")
    print(f"Prediksi        : {label.upper()}")


img_path_input = input("Masukan path ke gambar kopi yang ingin di prediksi: ")
prediksi_kualitas(img_path_input)