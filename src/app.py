"""Compatibility alias for the web application."""
import sys
if __name__ == '__main__':
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.web import app

if __name__ == '__main__':
    app.build_app().launch(theme=app.theme)
else:
    sys.modules[__name__] = app
