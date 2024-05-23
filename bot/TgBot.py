from datetime import datetime, timedelta
from cgitb import text
from email import message
from email.policy import default
import telebot
from telebot import types
import os
import time
import random
import cv2

allowed = [
    "701143942",
    "808250723"
]

bot = telebot.TeleBot('7058821742:AAFF-CXycpTOK8JNMNTK8sIRlJPjAWxuI2A')
animals_dir = "photo/photo_network"
cam_dir = "photo"
images = os.listdir(animals_dir)
for i in range(len(images)):
    print(images[i])

photos_num = len(images) - 1
streaming = True
learning = False
improving = False
names = ["котик",
         "песик",
         "хомячок"
         ]

names_num = len(names) - 1


@bot.message_handler(commands=['start', 'stop'])
def start(message):
    if (is_allowed(message)):
        bot.send_message(message.chat.id, text="Доступ разрешен")
        global streaming
        global learning
        global improving
        streaming = True
        learning = False
        improving = False
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)  # наша клавиатура
        key_start_learning = types.KeyboardButton('Начать обучение')  # кнопка «Начать обучение»
        key_start_correction = types.KeyboardButton('Улучшить точность')  # кнопка «Начать обучение»
        key_show_report = types.KeyboardButton('Показать отчет за последние сутки')
        keyboard.add(key_start_learning, key_start_correction, key_show_report)
        bot.send_message(message.chat.id, text="Режим дежурства", reply_markup=keyboard)
        check(message)
        os.chdir("photo")
        while streaming:
            for filename in os.listdir():
                current_time = time.time()
                creation_time = os.path.getmtime(filename)
                if (filename.endswith(".jpg")) and (current_time - creation_time < 15):
                    bot.send_photo(message.chat.id, filename)
                    bot.send_message(message.chat.id, text="Обнаружен человек")
    else:
        bot.send_message(message.chat.id, text="Доступ запрещен")


@bot.message_handler(commands=['check'])
def check(message):
    if (is_allowed(message)):
        bot.send_message(message.chat.id, text="Доступ разрешен")
        # Включаем первую камеру
        cap = cv2.VideoCapture(0)
        # "Прогреваем" камеру, чтобы снимок не был тёмным
        for i in range(30):
            cap.read()
        # Делаем снимок
        ret, frame = cap.read()
        # Записываем в файл
        cv2.imwrite(cam_dir + 'cam.png', frame)
        # Отключаем камеру
        cap.release()
        bot.send_photo(message.chat.id, open(cam_dir + "cam.png", 'rb'))

    else:
        bot.send_message(message.chat.id, text="Доступ запрещен")


@bot.message_handler(commands=['learn'])
def learn(message):
    if (is_allowed(message)):
        bot.send_message(message.chat.id, text="Доступ разрешен")
        global streaming
        global learning
        streaming = False
        learning = True
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        back_btn = types.KeyboardButton("Закончить обучение")
        markup.add(back_btn)
        bot.send_message(message.chat.id, text="Режим обучения", reply_markup=markup)
        i = random.randint(0, photos_num)
        j = random.randint(0, names_num)
        bot.send_photo(message.chat.id, open(animals_dir + "\\" + images[i], 'rb'))
        bot.send_message(message.chat.id, text="Кто это?".format(names[j]))

    else:
        bot.send_message(message.chat.id, text="Доступ запрещен")


@bot.message_handler(commands=['improvement'])
def improve(message):
    if (is_allowed(message)):
        bot.send_message(message.chat.id, text="Доступ разрешен")
        global streaming
        global improving
        streaming = False
        improving = True
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        end_btn = types.KeyboardButton("Закончить улучшение точности")
        yes_btn = types.KeyboardButton("Да")
        no_btn = types.KeyboardButton("Нет")
        skip_btn = types.KeyboardButton("Нет человека")
        markup.add(yes_btn, no_btn, skip_btn, end_btn)
        bot.send_message(message.chat.id, text="Режим улучшения точности", reply_markup=markup)
        i = random.randint(0, photos_num)
        j = random.randint(0, names_num)
        bot.send_photo(message.chat.id, open(animals_dir + "\\" + images[i], 'rb'))
        bot.send_message(message.chat.id, text="Это {}?".format(names[j]))

    else:
        bot.send_message(message.chat.id, text="Доступ запрещен")


@bot.message_handler(commands=['report'])
def report(message):
    if (streaming == True):
        date = datetime.now().date() - timedelta(days=1)
        bot.send_message(message.chat.id, text=str(date) + report_message)
    else:
        wrong_command(message)


@bot.message_handler(commands=['yes'])
def yes(message):
    if (improving == True):
        bot.send_message(message.chat.id, text="Супер")
        i = random.randint(0, photos_num)
        j = random.randint(0, names_num)
        bot.send_photo(message.chat.id, open(animals_dir + "\\" + images[i], 'rb'))
        bot.send_message(message.chat.id, text="Это {}?".format(names[j]))
    else:
        wrong_command(message)


@bot.message_handler(commands=['no'])
def no(message):
    if (improving == True):
        bot.send_message(message.chat.id, text="Принято")
        i = random.randint(0, photos_num)
        j = random.randint(0, names_num)
        bot.send_photo(message.chat.id, open(animals_dir + "\\" + images[i], 'rb'))
        bot.send_message(message.chat.id, text="Это {}?".format(names[j]))
    else:
        wrong_command(message)


@bot.message_handler(commands=['skip'])
def skip(message):
    if (improving == True):
        bot.send_message(message.chat.id, text="Прошу прощения")
        i = random.randint(0, photos_num)
        j = random.randint(0, names_num)
        bot.send_photo(message.chat.id, open(animals_dir + "\\" + images[i], 'rb'))
        bot.send_message(message.chat.id, text="Это {}?".format(names[j]))
    else:
        wrong_command(message)


@bot.message_handler(commands=['help'])
def help(message):
    if (streaming == True):
        bot.send_message(message.chat.id,
                         text="/learn - начать обучение\n/improvement - начать улучшение точности\n/report - получить отчет за последние сутки\n/check - получить кадр с камеры")

    elif (learning == True):
        bot.send_message(message.chat.id, text="/stop - закончить обучение\n/check - получить кадр с камеры")

    else:
        bot.send_message(message.chat.id,
                         text="/yes - подтвердить идентификацию\n/no - опровергнуть идентификацию\n/skip - пропустить некорректное изображение\n/stop - закончить улучшение точности\n/check - получить кадр с камеры")


@bot.message_handler(content_types=['text'])
def func(message):
    if (message.text == "Улучшить точность"):
        improve(message)

    elif (message.text == "Начать обучение"):
        learn(message)

    elif (message.text in ["Закончить обучение", "Закончить улучшение точности"]):
        start(message)

    elif (message.text == "Да"):
        yes(message)

    elif (message.text == "Нет"):
        no(message)

    elif (message.text == "Нет человека"):
        skip(message)

    elif (message.text == 'Показать отчет за последние сутки'):
        report(message)

    elif (learning == True and message.text not in names):
        bot.send_message(message.chat.id, text="Неверный формат, введите существующее имя")

    elif (learning == True and message.text in names):
        bot.send_message(message.chat.id, text="Я запомню")
        i = random.randint(0, photos_num)
        bot.send_photo(message.chat.id, open(animals_dir + "\\" + images[i], 'rb'))
        bot.send_message(message.chat.id, text="Кто это?")

    else:
        wrong_command(message)


def stream(message):
    i = random.randint(0, photos_num)
    bot.send_photo(message.chat.id, open(animals_dir + "\\" + images[i], 'rb'))
    bot.send_message(message.chat.id, text="Обнаружено движение")


def wrong_command(message):
    bot.send_message(message.chat.id,
                     text="Неверная команда, для того, чтобы посмотреть список доступных команд напишите /help")


def is_allowed(message):
    print(message.chat.id)
    if (str(message.chat.id) in allowed):
        return True
    return False


bot.polling(none_stop=True, interval=0)
