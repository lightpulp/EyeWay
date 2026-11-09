import io
import queue
import threading
import time

import requests
from flask import Flask, Response, jsonify, render_template_string
from PIL import Image, ImageDraw

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

ESP32_CAPTURE_URL = "http://192.168.1.122/capture"
API_URL = "http://127.0.0.1:5000/predict"
CONF_THRESHOLD = 0.8      # alert threshold, same as before
ALERT_COOLDOWN = 3.0      # seconds before the same class is announced again
ALERT_DISPLAY_TIME = 3.0  # how long the alert banner stays on screen
WEB_HOST = "127.0.0.1"    # use "0.0.0.0" to view the UI from a phone on the same network
WEB_PORT = 8000

app = Flask(__name__)

state_lock = threading.Lock()
latest = {
    "frame": None,           # newest raw JPEG from the ESP32
    "predicted_class": None,
    "probability": 0.0,
    "status": "Starting...",
    "alert": "",
    "alert_time": 0.0,
    "log": [],
}

speech_queue = queue.Queue(maxsize=5)


def speech_worker():
    """Runs speech in one dedicated thread, since pyttsx3 isn't thread-safe."""
    if pyttsx3 is None:
        return
    engine = pyttsx3.init()
    while True:
        text = speech_queue.get()
        engine.say(text)
        engine.runAndWait()


def say(text):
    if pyttsx3 is None:
        return
    try:
        speech_queue.put_nowait(text)
    except queue.Full:
        pass


def capture_loop():
    """Continuously grabs the newest frame from the ESP32."""
    session = requests.Session()
    while True:
        try:
            r = session.get(ESP32_CAPTURE_URL, timeout=5)
            r.raise_for_status()
            with state_lock:
                latest["frame"] = r.content
        except requests.RequestException as e:
            with state_lock:
                latest["status"] = f"Camera error: {e}"
            time.sleep(1)


def predict_loop():
    """Sends the newest frame to the Flask API and handles alerts."""
    session = requests.Session()
    last_alert_class = None
    last_alert_time = 0.0

    while True:
        with state_lock:
            frame = latest["frame"]
        if frame is None:
            time.sleep(0.2)
            continue

        try:
            r = session.post(
                API_URL,
                files={"image": ("frame.jpg", frame, "image/jpeg")},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            cls = data["predicted_class"]
            prob = float(data["predicted_probability"])
        except (requests.RequestException, KeyError, ValueError) as e:
            with state_lock:
                latest["status"] = f"API error: {e}"
            time.sleep(1)
            continue

        now = time.time()
        new_alert = None
        if prob >= CONF_THRESHOLD:
            if cls != last_alert_class or now - last_alert_time > ALERT_COOLDOWN:
                new_alert = f"A '{cls}' has been detected!!"
                say(new_alert)
                last_alert_class = cls
                last_alert_time = now

        with state_lock:
            latest["predicted_class"] = cls
            latest["probability"] = prob
            latest["status"] = "Running"
            if new_alert:
                latest["alert"] = new_alert
                latest["alert_time"] = now
                latest["log"].insert(0, f"{time.strftime('%H:%M:%S')}  {new_alert}")
                del latest["log"][20:]


def annotate(jpeg_bytes, label, prob):
    """Draws the current prediction on top of a camera frame."""
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    if label:
        color = (0, 220, 0) if prob >= CONF_THRESHOLD else (255, 180, 0)
        text = f"{label}  {prob * 100:.0f}%"
    else:
        color = (255, 255, 255)
        text = "waiting for prediction..."
    draw.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    draw.text((8, 9), text, fill=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video")
def video():
    def generate():
        while True:
            with state_lock:
                frame = latest["frame"]
                label = latest["predicted_class"]
                prob = latest["probability"]
            if frame is None:
                time.sleep(0.1)
                continue
            jpg = annotate(frame, label, prob)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        alert = latest["alert"] if time.time() - latest["alert_time"] < ALERT_DISPLAY_TIME else ""
        return jsonify({
            "predicted_class": latest["predicted_class"],
            "probability": latest["probability"],
            "status": latest["status"],
            "alert": alert,
            "log": list(latest["log"]),
        })


HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Obstacle Detection Live</title>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }
  h1 { margin: 0 0 16px; font-size: 22px; }
  .wrap { display: flex; gap: 20px; flex-wrap: wrap; }
  .feed { flex: 2 1 480px; }
  .feed img { width: 100%; border-radius: 8px; background: #000; }
  .side { flex: 1 1 260px; }
  .card { background: #1c1c1c; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .label { font-size: 34px; font-weight: 700; color: #6f6; }
  .muted { color: #999; font-size: 14px; }
  .alert { padding: 16px; border-radius: 8px; background: #222; color: #888; font-size: 20px; font-weight: 600; text-align: center; min-height: 24px; }
  .alert.on { background: #b00020; color: #fff; }
  ul { padding-left: 18px; margin: 0; }
  li { margin-bottom: 6px; font-size: 14px; }
</style>
</head>
<body>
<h1>Obstacle Detection, live</h1>
<div class="wrap">
  <div class="feed">
    <img src="/video" alt="Live camera feed">
  </div>
  <div class="side">
    <div class="card">
      <div class="muted">Detected</div>
      <div class="label" id="label">-</div>
      <div class="muted" id="prob"></div>
    </div>
    <div class="card">
      <div id="alert" class="alert">No alerts</div>
    </div>
    <div class="card">
      <div class="muted" id="status">Connecting...</div>
    </div>
    <div class="card">
      <div class="muted" style="margin-bottom:8px">Recent alerts</div>
      <ul id="log"></ul>
    </div>
  </div>
</div>
<script>
async function refresh() {
  try {
    const r = await fetch('/status');
    const s = await r.json();
    document.getElementById('label').textContent = s.predicted_class || '-';
    document.getElementById('prob').textContent = s.predicted_class ? (s.probability * 100).toFixed(0) + '% confidence' : '';
    document.getElementById('status').textContent = s.status;
    const alertEl = document.getElementById('alert');
    alertEl.textContent = s.alert || 'No alerts';
    alertEl.className = s.alert ? 'alert on' : 'alert';
    const items = s.log.map(text => {
      const li = document.createElement('li');
      li.textContent = text;
      return li;
    });
    document.getElementById('log').replaceChildren(...items);
  } catch (e) {
    document.getElementById('status').textContent = 'UI cannot reach server';
  }
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=predict_loop, daemon=True).start()
    threading.Thread(target=speech_worker, daemon=True).start()
    # debug=False matters here: the debug reloader would start the background threads twice
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, debug=False)