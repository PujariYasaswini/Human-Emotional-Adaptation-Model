import cv2
import threading
import time

class WebcamStream:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps

        # Open webcam
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        # Set width, height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        # Start background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """ Continuously grab frames from camera """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

            # Control FPS
            if self.fps > 0:
                time.sleep(1 / self.fps)

    def get_frame(self):
        """ Always return latest frame """
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self):
        """ Stop thread and release camera """
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()
