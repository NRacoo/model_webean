"""
ws_server.py — WebSocket Server untuk data deteksi YOLO
Jalankan dulu: pip install websockets
Lalu: python ws_server.py
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Simpan semua client yang terhubung ───────────────────
connected_clients   = set()   # browser / dashboard
detector_client     = None    # model_detect.py

# ── Statistik sesi ───────────────────────────────────────
session_stats = {
    "total_baik"  : 0,
    "total_cacat" : 0,
    "frames"      : 0,
    "started_at"  : None,
}

async def broadcast_to_clients(message: dict):
    """Kirim pesan ke semua dashboard client (browser)."""
    if not connected_clients:
        return
    data = json.dumps(message, ensure_ascii=False)
    await asyncio.gather(
        *[client.send(data) for client in connected_clients],
        return_exceptions=True
    )

async def handler(websocket):
    global detector_client

    # Identifikasi tipe koneksi dari header atau pesan pertama
    remote = websocket.remote_address
    log.info(f'Koneksi baru dari {remote}')

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f'Pesan bukan JSON: {raw[:80]}')
                continue

            msg_type = msg.get('type', '')

            # ── Pendaftaran client ────────────────────────
            if msg_type == 'register':
                role = msg.get('role', 'dashboard')

                if role == 'detector':
                    detector_client = websocket
                    session_stats['started_at'] = datetime.now().isoformat()
                    session_stats['total_baik']  = 0
                    session_stats['total_cacat'] = 0
                    session_stats['frames']      = 0
                    log.info('Detector terdaftar — sesi baru dimulai.')
                    await websocket.send(json.dumps({"type": "registered", "role": "detector"}))

                elif role == 'dashboard':
                    connected_clients.add(websocket)
                    log.info(f'Dashboard terdaftar. Total dashboard: {len(connected_clients)}')
                    # Kirim stats sesi saat ini ke dashboard baru
                    await websocket.send(json.dumps({
                        "type"        : "session_stats",
                        "total_baik"  : session_stats["total_baik"],
                        "total_cacat" : session_stats["total_cacat"],
                        "frames"      : session_stats["frames"],
                        "started_at"  : session_stats["started_at"],
                    }))

            # ── Data deteksi dari model_detect.py ─────────
            elif msg_type == 'detection':
                baik  = msg.get('baik',  0)
                cacat = msg.get('cacat', 0)
                fps   = msg.get('fps',   0.0)
                frame = msg.get('frame', 0)

                # Update statistik sesi
                session_stats['total_baik']  += baik
                session_stats['total_cacat'] += cacat
                session_stats['frames']       = frame

                payload = {
                    "type"        : "detection",
                    "baik"        : baik,
                    "cacat"       : cacat,
                    "fps"         : round(fps, 1),
                    "frame"       : frame,
                    "total_baik"  : session_stats["total_baik"],
                    "total_cacat" : session_stats["total_cacat"],
                    "timestamp"   : datetime.now().isoformat(),
                }

                log.info(f'Frame {frame} | Baik: {baik} Cacat: {cacat} | '
                         f'Total Baik: {session_stats["total_baik"]} '
                         f'Total Cacat: {session_stats["total_cacat"]} | FPS: {fps:.1f}')

                await broadcast_to_clients(payload)

            # ── Ping keepalive ────────────────────────────
            elif msg_type == 'ping':
                await websocket.send(json.dumps({"type": "pong"}))

            else:
                log.warning(f'Tipe pesan tidak dikenal: {msg_type}')

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning(f'Koneksi terputus: {e}')
    finally:
        # Cleanup saat client disconnect
        if websocket == detector_client:
            detector_client = None
            log.info('Detector terputus.')
            await broadcast_to_clients({"type": "detector_disconnected"})
        connected_clients.discard(websocket)
        log.info(f'Koneksi {remote} ditutup. Dashboard aktif: {len(connected_clients)}')

async def main():
    HOST = '0.0.0.0'
    PORT = 8765
    log.info(f'WebSocket server berjalan di ws://{HOST}:{PORT}')
    log.info('Menunggu koneksi dari detector dan dashboard...')

    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()  # jalan selamanya

if __name__ == '__main__':
    asyncio.run(main())