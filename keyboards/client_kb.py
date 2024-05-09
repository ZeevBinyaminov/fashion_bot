from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, \
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove

from loader import bot
from aiogram import types

# start menu keyboard
help_button = KeyboardButton('/help')
main_menu = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
main_menu.add(help_button)


# inline keyboard markup

special_event_button = InlineKeyboardButton(text="Подобрать образ к особому событию (вне MVP)",
                                            callback_data='special event')
new_outfits_button = InlineKeyboardButton(text="Подобрать новые образы",
                                          callback_data='new outfits')
wardrobe_review_button = InlineKeyboardButton(text="Сделать разбор гардероба (вне MVP)",
                                              callback_data='wardrobe review')
cancel_button = InlineKeyboardButton(text="Отмена",
                                     callback_data='cancel')

cancel_inkb = InlineKeyboardMarkup().add(cancel_button)


# Клавиатуры для inline-кнопок
silhouette_keyboard = types.InlineKeyboardMarkup(row_width=1)
silhouette_keyboard.add(
    types.InlineKeyboardButton(
        text="Облегающий силуэт с мягкими изгибами", callback_data='q1_1'),
    types.InlineKeyboardButton(
        text="Свободный силуэт с минимум украшений", callback_data='q1_2'),
    types.InlineKeyboardButton(
        text="Прямой и строгий силуэт с острыми линиями", callback_data='q1_3'),
    types.InlineKeyboardButton(
        text="Сбалансированный силуэт с четкими линиями", callback_data='q1_4'),
    types.InlineKeyboardButton(
        text="Динамичный силуэт с сочетанием острых и мягких линий", callback_data='q1_5')
)

materials_keyboard = types.InlineKeyboardMarkup(row_width=1)
materials_keyboard.add(
    types.InlineKeyboardButton(
        text="Четкие, жесткие ткани, подчеркивающие форму", callback_data='q2_1'),
    types.InlineKeyboardButton(
        text="Мягкие, текучие ткани с декоративными деталями", callback_data='q2_2'),
    types.InlineKeyboardButton(
        text="Умеренно мягкие ткани с минимальной детализацией", callback_data='q2_3'),
    types.InlineKeyboardButton(
        text="Натуральные, легкие ткани для максимального комфорта", callback_data='q2_4'),
    types.InlineKeyboardButton(
        text="Смешанные ткани с контрастными", callback_data='q2_5')
)

colors_keyboard = types.InlineKeyboardMarkup(row_width=1)
colors_keyboard.add(
    types.InlineKeyboardButton(
        text="Монохромные цвета и геометрические узоры", callback_data='q3_1'),
    types.InlineKeyboardButton(
        text="Нейтральные цвета и классические узоры", callback_data='q3_2'),
    types.InlineKeyboardButton(
        text="Яркие цвета и контрастные комбинации", callback_data='q3_3'),
    types.InlineKeyboardButton(
        text="Пастельные оттенки и цветочные узоры", callback_data='q3_4'),
    types.InlineKeyboardButton(
        text="Землистые тон, без узоров или с минималистичными узорами", callback_data='q3_5')
)
