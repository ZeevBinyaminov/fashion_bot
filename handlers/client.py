
import virtual_stylist_copy
from database.database import user_db, photo_db
from loader import bot, dp
import os
import re
import io
from pathlib import Path

from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher.filters import Text


from keyboards.client_kb import (silhouette_keyboard, materials_keyboard,
                                 colors_keyboard, main_inkb, cancel_inkb)
# Функция для сохранения ответа в базе данных


class States(StatesGroup):
    Q1 = State()
    Q2 = State()
    Q3 = State()
    HEIGHT = State()
    SHOULDER_WIDTH = State()
    WAIST_CIRCUMFERENCE = State()
    CHEST_CIRCUMFERENCE = State()
    HIP_CIRCUMFERENCE = State()
    PHOTO = State()


@dp.message_handler(commands=['start'])
async def start_survey(message: types.Message, state: FSMContext):
    await message.answer("Добро пожаловать в ваш Цифровой Помощник по Гардеробу. Чем могу помочь сегодня?",
                         reply_markup=main_inkb)


# Обработчики для стартовых опций
@dp.callback_query_handler(lambda callback_query: callback_query.data in ["special_event_outfit", "new_outfits", "wardrobe_review"])
async def handle_start_options(callback_query: types.CallbackQuery, state: FSMContext):
    option = callback_query.data

    if option == "special_event_outfit":
        # await callback_query.message.answer("Опция 'Подобрать образ к особому событию' находится вне MVP.")
        return
    elif option == "wardrobe_review":
        # await callback_query.message.answer("Опция 'Сделать разбор гардероба' находится вне MVP.")
        return

    # Опция "Подобрать новые образы"
    await callback_query.message.answer("Выберите, какой силуэт одежды вам наиболее нравится:", reply_markup=silhouette_keyboard)
    await States.Q1.set()
    await bot.delete_message(callback_query.from_user.id, callback_query.message.message_id)


@dp.callback_query_handler(state='*', regexp='q1_.*|q2_.*|q3_.*')
async def process_answers(callback_query: types.CallbackQuery, state: FSMContext):
    async with state.proxy() as data:
        if callback_query.data.startswith('q1_'):
            data['silhouette'] = callback_query.data[3:]
            await bot.edit_message_text("Отлично! Теперь выберите предпочитаемые материалы в одежде:",
                                        callback_query.from_user.id, callback_query.message.message_id,
                                        reply_markup=materials_keyboard)
            await States.Q2.set()
        elif callback_query.data.startswith('q2_'):
            data['materials'] = callback_query.data[3:]
            await bot.edit_message_text("Отлично! Теперь выберите предпочтительные цвета и узоры в одежде:",
                                        callback_query.from_user.id, callback_query.message.message_id,
                                        reply_markup=colors_keyboard)
            await States.Q3.set()
        elif callback_query.data.startswith('q3_'):
            data['colors'] = callback_query.data[3:]
            await bot.edit_message_text("Хорошо! Чтобы точно подобрать образы, подходящие именно вам, "
                                        "ввдетие параметры Вашего тела.\n"
                                        "Пожалуйста, укажите ваш рост в сантиметрах:",
                                        callback_query.from_user.id, callback_query.message.message_id,
                                        )
            await States.HEIGHT.set()

# Обработка фото


@dp.message_handler(content_types=types.ContentType.PHOTO, state=States.PHOTO)
async def process_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]  # Берем последнее (самое большое) фото
    if photo.file_size > 10 * 1024 * 1024:  # Проверка на размер файла
        await message.answer("Файл слишком большой. Пожалуйста, загрузите файл размером менее 10 МБ.")
        return

    async with state.proxy() as data:
        data['photo'] = photo.file_id

    await user_db.save_survey_answers(message.from_user.id, data)
    filename = await virtual_stylist_copy.download_image(bot, data['photo'], message.from_user.id)

    shape_predictor = "shape_predictor_68_face_landmarks.dat"

    virtual_stylist_copy.faceAligned(filename, shape_predictor)
    image, face_type = await virtual_stylist_copy.test_typing(filename, shape_predictor, data['height'],
                                                              data['shoulder_width'], data['chest_circumference'],
                                                              data['waist_circumference'], data['hip_circumference'])
    face_type = face_type.split()[0]
    with open(image, 'rb') as parsed_photo:
        await message.answer_photo(photo=parsed_photo, caption=f"Образы соответствующие вашему типажу: {face_type}")

    folder = 'four_types'
    photos = photo_db.get_random_photos_by_face_type(face_type=face_type)
    for photo in photos:
        await message.answer_photo(photo=io.BytesIO(photo[0]))
    with open(list(Path(folder, face_type).glob('*.docx'))[0], 'rb') as description_file:
        await message.answer_document(description_file)

    os.remove(filename)
    os.remove(image)

    await state.finish()


# Функция для проверки числового значения

def is_valid_number(text):
    return bool(re.match(r'^\d+$', text))

# Обработка роста


@dp.message_handler(state=States.HEIGHT)
async def process_height(message: types.Message, state: FSMContext):
    if not is_valid_number(message.text):
        await message.answer("Некорректный формат. Пожалуйста, укажите ваш рост в сантиметрах (только числом):")
        return

    async with state.proxy() as data:
        data['height'] = int(message.text)
    await message.answer("Теперь укажите ширину ваших плеч в сантиметрах:", )
    await States.SHOULDER_WIDTH.set()

# Обработка ширины плеч


@dp.message_handler(state=States.SHOULDER_WIDTH)
async def process_shoulder_width(message: types.Message, state: FSMContext):
    if not is_valid_number(message.text):
        await message.answer("Некорректный формат. Пожалуйста, укажите ширину ваших плеч в сантиметрах (только числом):")
        return

    async with state.proxy() as data:
        data['shoulder_width'] = int(message.text)
    await message.answer("Теперь укажите окружность вашей талии в сантиметрах:",
                         )
    await States.WAIST_CIRCUMFERENCE.set()

# Обработка окружности талии


@dp.message_handler(state=States.WAIST_CIRCUMFERENCE)
async def process_waist_circumference(message: types.Message, state: FSMContext):
    if not is_valid_number(message.text):
        await message.answer("Некорректный формат. Пожалуйста, укажите окружность вашей талии в сантиметрах (только числом):")
        return

    async with state.proxy() as data:
        data['waist_circumference'] = int(message.text)
    await message.answer("Теперь укажите обхват вашей груди в сантиметрах:",
                         )
    await States.CHEST_CIRCUMFERENCE.set()

# Обработка обхвата груди


@dp.message_handler(state=States.CHEST_CIRCUMFERENCE)
async def process_chest_circumference(message: types.Message, state: FSMContext):
    if not is_valid_number(message.text):
        await message.answer("Некорректный формат. Пожалуйста, укажите обхват вашей груди в сантиметрах (только числом):")
        return

    async with state.proxy() as data:
        data['chest_circumference'] = int(message.text)
    await message.answer("И последнее, укажите обхват ваших бедер в сантиметрах:",
                         )
    await States.HIP_CIRCUMFERENCE.set()


# Обработка обхвата бедер и отправка результатов


@dp.message_handler(state=States.HIP_CIRCUMFERENCE)
async def process_hip_circumference(message: types.Message, state: FSMContext):
    if not is_valid_number(message.text):
        await message.answer("Некорректный формат. Пожалуйста, укажите обхват ваших бедер в сантиметрах (только числом):")
        return

    async with state.proxy() as data:
        data['hip_circumference'] = int(message.text)

    await message.answer("Хорошо! Чтобы точно подобрать образы, подходящие именно Вам,"
                         "загрузите фотографию Вашего лица в анфас.",
                         )
    await States.PHOTO.set()


@dp.callback_query_handler(Text(equals='Отмена', ignore_case=True), state="*")
async def cancel_handler(message: types.Message, state=FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return
    await state.finish()
    await message.reply(reply_markup=main_inkb)
