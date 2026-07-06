"""
backend/app.py
-----------------
Flask application entrypoint for the AI-Enabled Fraud Detection backend.

RUN:
    python backend/app.py

This serves:
  - The REST API (blueprints registered below, all under /api/...)
  - The frontend (index.html + static assets) at the root URL "/"
"""

import os
import sys

from flask import Flask, render_template, send_from_directory

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

# Allow `python backend/app.py` to work regardless of the current working
# directory by ensuring the project root is on sys.path for absolute imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.routes.predict_routes import predict_bp
from backend.routes.dashboard_routes import dashboard_bp
from backend.utils.model_loader import fraud_model, ModelNotTrainedError
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def create_app():
    """Application factory - lets tests build an app instance independently."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    if CORS_AVAILABLE:
        CORS(app)  # allow the frontend (potentially served separately) to call the API
    else:
        logger.warning("flask_cors not installed - CORS disabled. Run: pip install flask-cors")

    app.register_blueprint(predict_bp)
    app.register_blueprint(dashboard_bp)

    @app.route("/")
    def index():
        """Serve the frontend single-page dashboard."""
        return render_template("index.html")

    @app.route("/health")
    def health():
        """Simple health check endpoint, useful for uptime checks / deployment probes."""
        model_ready = os.path.exists(config.BEST_MODEL_PATH)
        return {"status": "ok", "model_ready": model_ready}, 200

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Endpoint not found"}, 404

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("Unhandled server error")
        return {"error": "Internal server error"}, 500

    # Try to load the model eagerly at startup so the FIRST API request
    # doesn't pay the disk-load cost, and so a missing model is reported
    # clearly in the startup logs rather than surfacing later as a 503.
    try:
        fraud_model.load()
    except ModelNotTrainedError as e:
        logger.warning(f"{e} The server will start, but /api/predict will return 503 "
                        f"until you run the training script.")

    return app


app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting Flask server on {config.HOST}:{config.PORT} (debug={config.DEBUG})")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
