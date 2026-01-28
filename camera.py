import cv2
import numpy as np 
import tensorflow as tf
import time

modelPath = 'model_kopi.tflite'
imgH, imgW = 224, 224

class_labels = ['biji bagus', 'biji defek']

interpreter = tf.lite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

is_input_quantized = input_details[0]['dtype'] == np.uint8

print("[INFO] Model TFlite berhasil")
print(f"[INFO] Input Shape: {input_shape}")
print(f"[INFO] Input Type: {input_details[0]['dtype']}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Tidak bisa membuka kamera!")
    exit()
print("[INFO] Kamera menyala")

fps_start_time = time.time()
fps_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Gagal mengambil frame")
    
    display_frame = frame.copy()

    img_resized = cv2.resize(frame, (imgW, imgH))

    input_data = np.expand_dims(img_resized, axis=0)

    if not is_input_quantized:
        input_data = (input_data.astype(np.float32)) / 255.0
    
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if prediction > 0.5:
        label = class_labels[1]
        confidence = 1 - prediction
        color = (0, 0, 255)
    else:
        label = class_labels[0]
        confidence = 1-prediction
        color = (0, 255, 0)

    if label == 'biji_defek':
        print("[DEFECT DETECTED]")
        pass #GPIO KODE
    else:
        pass #GPIO KODE
    
    fps_counter += 1
    if time.time() - fps_start_time >=1.0:
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
        print(f"[INFO] FPS: {fps}")
    
    text = f"{label.upper} ({confidence*100:.1f})"
    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

    cv2.imshow('Penyortiran-Kopi', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Menutup program")
cap.release()
cv2.destroyAllWindows()
