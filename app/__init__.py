import os
from flask import Flask
from dotenv import load_dotenv

def create_app():
    load_dotenv()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
    app.config["DATA_DIR"] = os.path.join(os.path.dirname(__file__), "..", "data")
    app.config["UPLOADS_DIR"] = os.path.join(app.config["DATA_DIR"], "uploads")
    app.config["JOBS_DIR"] = os.path.join(app.config["DATA_DIR"], "jobs")
    app.config["DB_PATH"] = os.path.join(app.config["DATA_DIR"], "app.db")
    app.config["TAXONOMY_TEMPLATE_CSV"] = os.environ.get(
        "TAXONOMY_TEMPLATE_CSV",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "taxonomy_template.csv")),
    )

    os.makedirs(app.config["UPLOADS_DIR"], exist_ok=True)
    os.makedirs(app.config["JOBS_DIR"], exist_ok=True)

    from .db import init_db
    init_db(app.config["DB_PATH"])

    from .routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app
