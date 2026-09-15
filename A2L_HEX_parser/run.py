import os, sys, threading, webbrowser
import streamlit.web.cli as stcli

# Ensure the bundle folder is importable at runtime (so app.py's sibling imports resolve)
if hasattr(sys, "_MEIPASS"):
    sys.path.insert(0, sys._MEIPASS)

def resolve(path):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, path)

# Force PyInstaller to bundle these and trace their dependencies
import pipeline
import a2l_hex_to_cdfx
import cdfx_to_a2l_hex

if __name__ == "__main__":
    threading.Timer(3, lambda: webbrowser.open("http://localhost:8501")).start()
    sys.argv = [
        "streamlit", "run", resolve("app.py"),
        "--global.developmentMode=false",
        "--server.maxUploadSize=10000",
        "--server.enableXsrfProtection=false",
    ]
#     sys.argv = [
#     "streamlit", "run", resolve("app.py"),
#     "--global.developmentMode=false",
#     "--server.headless=true",
# ]
    sys.exit(stcli.main())