import traceback

try:
    import streamlit.web.cli as stcli
    import sys
    import os

    print("Starting launcher")

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print("App path:", app_path)

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.maxUploadSize=10000",
        "--server.enableXsrfProtection=false",
    ]

    stcli.main()

except Exception:
    traceback.print_exc()
    input("Press Enter to exit...")