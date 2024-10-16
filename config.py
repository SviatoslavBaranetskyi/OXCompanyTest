import os
from datetime import timedelta


class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:dbpass@localhost/anime_analysis')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt_secret_key')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
    AES_KEY = os.getenv('AES_KEY', os.urandom(32))
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads/images')
