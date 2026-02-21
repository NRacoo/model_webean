import os
import sys
import argparse
import glob
import time
import threading
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════
#  KONFIGURASI LABEL BAIK / CACAT
# ══════════════════════════════════════════════════════════
LABEL_BAIK  = ['baik', 'good', 'ok']
LABEL_CACAT = ['cacat', 'defect', 'bad', 'rusak']

# ══════════════════════════════════════════════════════════
#  KONFIGURASI OPTIMASI
# ══════════════════════════════════════════════════════════
INFER_SIZE  = 320   # input model: 320 lebih cepat dari 640
SKIP_FRAMES = 2     # inferensi setiap N+1 frame (2 = proses 1, skip 2)
# ══════════════════════════════════════════════════════════


SIO_ENABLED = True
SIO_URL     = 'http://localhost:3000'
SIO_NAMESPACE = '/detection'

parser = argparse.ArgumentParser()
parser.add_argument('--model',       required=True)
parser.add_argument('--source',      required=True,
                    help='usb0 / usb1 / /dev/video0 / 0 / video.mp4 / folder/')
parser.add_argument('--thresh',      default=0.5)
parser.add_argument('--resolution',  default=None, help='WxH contoh: 640x480')
parser.add_argument('--record',      action='store_true')
parser.add_argument('--export-ncnn', action='store_true')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res   = args.resolution
record     = args.record

# ── Export NCNN ──────────────────────────────────────────
if args.export_ncnn:
    print('[INFO] Mengexport model ke NCNN...')
    tmp = YOLO(model_path)
    tmp.export(format='ncnn', imgsz=INFER_SIZE)
    print('[INFO] Selesai. Gunakan folder *_ncnn_model sebagai --model.')
    sys.exit(0)

# ── Validasi model ───────────────────────────────────────
if not os.path.exists(model_path):
    print('ERROR: Model tidak ditemukan.')
    sys.exit(0)

# ── Load model ───────────────────────────────────────────
print(f'[INFO] Loading model: {model_path}')
model  = YOLO(model_path, task='detect')
labels = model.names


sio = None
sio_ready = False

def start_socketio():
    global sio, sio_ready
    try:
        import socketio as sio_lib
    except ImportError:
        print('[SIO] ERROR: install dulu:')
        print('      pip install "python-socketio[client]" websocket-client')
        return
    sio = sio_lib.Client(
        reconnection=True,
        reconnection_attempt=0,
        reconnection_delay=3,
        reconnection_delay_max=30,
        logger=False,
        engineio_logger=False
    )
    
    @sio.event
    def connect():
        global sio_ready
        sio_ready=True
        print(f'[SIO] Terhubung ke NestJS {SIO_URL}-{SIO_NAMESPACE}')
        sio.emit('register', {'role':'detector'})
    
    @sio.event
    def disconnect():
        global sio_ready
        sio_ready=False
        print('[SIO] Terputus dari NestJS. Reconnecting ...')
    
    @sio.on('detector_status')
    def on_status(data):
        print(f'[SIO] Status dari server: {data}')
        try:
            sio.connect(
                SIO_URL,
                namespaces=[SIO_NAMESPACE],
                socketio_path='/socket.io',
                transports=['websocket']
            )
            sio.wait()
        except Exception as e :
            print(f'[SIO] Gagal konek: {e}. Retry dalam 5s')
            time.sleep(5)

def send_detection(baik:int, cacat:int, fps:float, frame_num: int):
    if not SIO_ENABLED or not sio_ready or sio is None:
        return
    try:
        sio.emit('detection', {
            'baik':baik,
            'cacat':cacat,
            'fps':fps,
            'frame':frame_num
        })
    except Exception as e:
        print(f'[SIO] Gagal kirim: {e}')

if SIO_ENABLED:
    sio_thread = threading.Thread(target=start_socketio, daemon=True)
    sio_thread.start()
    print('[SIO] Thread Socket.IO dimulai.')

# ── Parse source — support berbagai format ────────────────
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

source_type = None
usb_idx     = None

if os.path.isdir(img_source):
    source_type = 'folder'

elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Ekstensi {ext} tidak didukung.')
        sys.exit(0)

elif img_source.startswith('/dev/video'):
    # Format: /dev/video0, /dev/video1, dst
    source_type = 'usb'
    usb_idx = int(img_source.replace('/dev/video', ''))

elif img_source.lower().startswith('usb'):
    # Format: usb0, usb1, dst
    source_type = 'usb'
    usb_idx = int(img_source[3:])

elif img_source.isdigit():
    # Format: 0, 1, 2, dst
    source_type = 'usb'
    usb_idx = int(img_source)

elif 'picamera' in img_source.lower():
    source_type = 'picamera'

else:
    print(f'Input "{img_source}" tidak valid.')
    print('Contoh yang valid: usb0, /dev/video0, 0, video.mp4, folder_gambar/')
    sys.exit(0)

print(f'[INFO] Source type: {source_type}, index: {usb_idx}')

# ── Resolusi ─────────────────────────────────────────────
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# ── Recording ────────────────────────────────────────────
if record:
    if source_type not in ['video', 'usb', 'picamera']:
        print('Record hanya untuk video/kamera.')
        sys.exit(0)
    if not user_res:
        print('Tentukan --resolution untuk recording.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# ── Init source ──────────────────────────────────────────
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*')
                 if os.path.splitext(f)[1] in img_ext_list]

elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    print(f'[INFO] Membuka kamera index: {cap_arg}')
    cap = cv2.VideoCapture(cap_arg)

    if not cap.isOpened():
        print(f'ERROR: Tidak bisa membuka kamera {cap_arg}')
        print('Tips: coba index lain (usb1) atau cek dengan: sudo fuser /dev/video0')
        sys.exit(0)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

    # Tunggu kamera siap & flush frame lama
    print('[INFO] Menunggu kamera siap...')
    time.sleep(2)
    for _ in range(5):
        cap.read()
    print('[INFO] Kamera siap.')

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    w = resW if user_res else 640
    h = resH if user_res else 480
    cap.configure(cap.create_video_configuration(
        main={"format": 'XRGB8888', "size": (w, h)}))
    cap.start()
    time.sleep(1)

# ── Warna bbox ───────────────────────────────────────────
bbox_colors = [(164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),
               (96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)]

# ── State ────────────────────────────────────────────────
avg_frame_rate   = 0
fps_buffer       = deque(maxlen=60)
img_count        = 0
frame_counter    = 0
last_detections  = []
last_count_baik  = 0
last_count_cacat = 0

print('[INFO] Memulai inferensi... Tekan Q untuk keluar.')

while True:
    t_start = time.perf_counter()

    # ── Ambil frame ──────────────────────────────────────
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Semua gambar telah diproses.')
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Video selesai.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Gagal membaca frame dari kamera.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Gagal membaca frame dari Picamera.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # ── Inferensi dengan frame skipping ──────────────────
    run_inference = (frame_counter % (SKIP_FRAMES + 1) == 0)
    frame_counter += 1

    if run_inference:
        results    = model(frame, imgsz=INFER_SIZE, verbose=False)
        detections = results[0].boxes

        last_detections  = []
        last_count_baik  = 0
        last_count_cacat = 0

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx  = int(detections[i].cls.item())
            classname = labels[classidx]
            conf      = detections[i].conf.item()

            if conf > min_thresh:
                last_detections.append((xmin, ymin, xmax, ymax, classname, conf, classidx))
                cname_lower = classname.lower()
                if any(lbl in cname_lower for lbl in LABEL_BAIK):
                    last_count_baik += 1
                elif any(lbl in cname_lower for lbl in LABEL_CACAT):
                    last_count_cacat += 1
        send_detection(last_count_baik, last_count_cacat, avg_frame_rate, frame_counter)

    # ── Gambar bbox ──────────────────────────────────────
    for (xmin, ymin, xmax, ymax, classname, conf, classidx) in last_detections:
        color = bbox_colors[classidx % 10]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = f'{classname}: {int(conf*100)}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(frame,
                      (xmin, label_ymin-labelSize[1]-10),
                      (xmin+labelSize[0], label_ymin+baseLine-10),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    

    # ── Overlay info ─────────────────────────────────────
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.1f}', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.putText(frame, f'Baik  : {last_count_baik}',  (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
    cv2.putText(frame, f'Cacat : {last_count_cacat}', (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    wait_ms = 1 if source_type not in ['image', 'folder'] else 0
    key = cv2.waitKey(wait_ms)
    if   key in [ord('q'), ord('Q')]: break
    elif key in [ord('s'), ord('S')]: cv2.waitKey()
    elif key in [ord('p'), ord('P')]: cv2.imwrite('capture.png', frame)

    # ── Hitung FPS ───────────────────────────────────────
    t_stop  = time.perf_counter()
    elapsed = t_stop - t_start
    if elapsed > 0:
        fps_buffer.append(1.0 / elapsed)
        avg_frame_rate = float(np.mean(fps_buffer))

# ── Cleanup ───────────────────────────────────────────────
print(f'Rata-rata FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()