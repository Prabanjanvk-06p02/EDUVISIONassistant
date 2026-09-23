import os

class Config:
    SECRET_KEY = "ai-smart-classroom-secret"
    DB_PATH = os.path.join("database", "attendance.db")
    DEBUG = True