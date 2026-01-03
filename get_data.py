import threading
import time
from typing import Optional

import cv2
from get_video_url import VideoTokenClient
from ultralytics import YOLO


class StreamDataClient:
    def __init__(
        self,
        stream_base_url: str = "https://edge01.balticlivecam.com/blc/narva/index.m3u8",
        token_source_url: str = "https://balticlivecam.com/cameras/estonia/narva/narva/",
        token_update_interval: int = 3600,
    ):
        self.stream_base_url = stream_base_url
        self.token_update_interval = token_update_interval
        self._token_client = VideoTokenClient(token_source_url)
        self._current_token: Optional[str] = None
        self._stop_event = threading.Event()

        self._refresh_token()
        self._start_refresh_loop()

    def _refresh_token(self) -> None:
        self._token_client.update_token()
        self._current_token = self._token_client.token

    def _start_refresh_loop(self) -> None:
        def _loop() -> None:
            while not self._stop_event.wait(self.token_update_interval):
                self._refresh_token()

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        self._refresh_thread = thread

    def get_stream_url(self) -> str:
        if not self._current_token:
            self._refresh_token()
        if not self._current_token:
            raise RuntimeError("Cannot obtain stream token.")
        return f"{self.stream_base_url}?token={self._current_token}"

    def close(self) -> None:
        self._stop_event.set()
        self._token_client.close()


def main():
    client = StreamDataClient()
    try:
        stream_url = client.get_stream_url()
        model = YOLO("yolo11n.pt")
        
        all_detections = []
        interval = 60 
        
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise RuntimeError("Cannot open stream")
        
        print(f"Analyzing stream every {interval} seconds...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue
            results = model.track(frame, persist=True, verbose=False)
            result = results[0]
            
            if result.boxes and result.boxes.id is not None:
                for box, track_id, cls, conf in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.id.int().cpu().tolist(),
                    result.boxes.cls.int().cpu().tolist(),
                    result.boxes.conf.cpu().tolist()
                ):
                    detection = {
                        "track_id": track_id,
                        "class_id": cls,
                        "class_name": model.names[cls],
                        "confidence": conf,
                        "bbox": box.tolist(), 
                    }
                    all_detections.append(detection)
                    print(f"Detected: {detection}")
            else:
                print("No objects detected")
            
            time.sleep(interval)
        
        print(f"\nTotal detections: {len(all_detections)}")
        print("All detections:", all_detections)
    finally:
        client.close()


if __name__ == "__main__":
    main()
