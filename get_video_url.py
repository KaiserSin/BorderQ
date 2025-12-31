import re
import time
from typing import Optional

from seleniumwire import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class VideoTokenClient:
    def __init__(self, target_url: str) -> None:
        self.target_url = target_url
        self.token: Optional[str] = None
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options,
        )

    def update_token(self) -> Optional[str]:
        self.token = None
        try:
            self.driver.requests.clear()
        except Exception:
            pass
        print(f"Opening {self.target_url}...")
        self.driver.get(self.target_url)
        time.sleep(2)

        for request in self.driver.requests:
            if request.response and "index.m3u8" in request.url and "token=" in request.url:
                match = re.search(r"token=([^&]+)", request.url)
                if match:
                    self.token = match.group(1)
                    break
        return self.token

    def close(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass
