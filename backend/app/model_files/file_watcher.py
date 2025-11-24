from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import time

# ŚCIEŻKA DO TWOJEGO CSV
CSV_PATH = Path(r"C:\Users\kolec\AppData\Roaming\MetaQuotes\Terminal\Common\Files\ai_forex\ohlc_EURUSD_H1.csv")

# plik sygnałowy dla streamlita
TRIGGER_FILE = Path("trigger_streamlit.txt")

class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        if Path(event.src_path) == CSV_PATH:
            print("CSV changed!")
            TRIGGER_FILE.write_text(str(time.time()))

observer = Observer()
observer.schedule(Handler(), CSV_PATH.parent, recursive=False)
observer.start()

print("Streamlit watcher running...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
