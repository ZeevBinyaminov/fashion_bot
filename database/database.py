import sqlite3
import os
from pathlib import Path
import random


class BaseDatabase:
    def __init__(self, filename, name='Params'):
        self.base = sqlite3.connect(filename)
        self.cursor = self.base.cursor()
        if self.base:
            print(f"{name} database connected")

    def close(self):
        self.base.close()


class UserDatabase(BaseDatabase):
    def create_database(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS "
                            "survey_answers ("
                            "user_id INTEGER NOT NULL,"
                            "silhouette TEXT,"
                            "materials TEXT,"
                            "colors TEXT,"
                            "photo BLOB,"
                            "height INTEGER,"
                            "shoulder_width INTEGER,"
                            "waist_circumference INTEGER,"
                            "chest_circumference INTEGER,"
                            "hip_circumference INTEGER)")

        self.base.commit()

    async def save_survey_answers(self, user_id, data):
        self.cursor.execute("""INSERT INTO survey_answers (user_id, silhouette, materials, colors, photo,
                                                           height, shoulder_width, waist_circumference, 
                                                           chest_circumference, hip_circumference)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (user_id,
                             data.get('silhouette'),
                             data.get('materials'),
                             data.get('colors'),
                             data.get('photo'),
                             data.get('height'),
                             data.get('shoulder_width'),
                             data.get('waist_circumference'),
                             data.get('chest_circumference'),
                             data.get('hip_circumference')))
        self.base.commit()


class PhotoDatabase(BaseDatabase):
    PHOTO_FOLDER = 'four_types'

    def __init__(self, filename):
        super().__init__(filename, name='Photo')

    def create_photo_table(self):
        flag = self.cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='photos'").fetchone()[0]
        self.cursor.execute("CREATE TABLE IF NOT EXISTS "
                            "photos ("
                            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                            "face_type TEXT,"
                            "photo BLOB)")

        self.base.commit()

        # Вставка фотографий при создании таблицы, если она только что была создана
        if not flag:
            self.insert_photos_from_folder(PhotoDatabase.PHOTO_FOLDER)

    def insert_photo(self, face_type, photo_data):
        self.cursor.execute(
            """INSERT INTO photos (face_type, photo) VALUES (?, ?)""", (face_type, photo_data))
        self.base.commit()

    def insert_photos_from_folder(self, folder):
        for face_type in os.listdir(folder):
            folder_path = os.path.join(folder, face_type)
            if os.path.isdir(folder_path):
                photos = [Path(folder_path, photo) for photo in os.listdir(
                    folder_path) if photo.endswith('.png')]

                # Вставка фотографий в базу данных
                for photo_path in photos:
                    with open(photo_path, 'rb') as photo_file:
                        photo_data = photo_file.read()
                        self.insert_photo(face_type, photo_data)

    def get_random_photos_by_face_type(self, face_type, quantity=5):
        self.cursor.execute(
            "SELECT photo FROM photos WHERE face_type = ?", (face_type,))
        photos = self.cursor.fetchall()
        random_photos = random.sample(photos, min(len(photos), quantity))
        return random_photos


# Создание экземпляра базы данных
user_db = UserDatabase("survey.db")
user_db.create_database()

# Создание экземпляра базы данных фотографий
photo_db = PhotoDatabase("photo.db")
photo_db.create_photo_table()
