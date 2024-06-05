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
import csv

allowed = [
    "701143942",
    "808250723"
]

bot = telebot.TeleBot('7058821742:AAFF-CXycpTOK8JNMNTK8sIRlJPjAWxuI2A')


faces_dir = "photo/photo_network"
cam_dir = "photo"
faces_time={}
access = False
streaming = True
learning = False
improving = False
names = []
names_num = len(names) - 1

@bot.message_handler(commands=['start', 'stop'])
def start(message):
    if (access):
        print("starting...")
        global streaming
        global learning
        global improving
        global faces_time
        streaming = True
        learning = False
        improving = False
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)  # наша клавиатура
        key_start_learning = types.KeyboardButton('Начать обучение')  # кнопка «Начать обучение»
        key_start_correction = types.KeyboardButton('Улучшить точность')  # кнопка «Начать обучение»
        key_show_report = types.KeyboardButton('Показать отчет за последние сутки')
        keyboard.add(key_start_learning, key_start_correction, key_show_report)
        bot.send_message(message.chat.id, text="Режим дежурства", reply_markup=keyboard)
        # os.chdir("photo/screenshot")
        # bot.send_photo(message.chat.id, open("", 'rb'))
        # os.chdir(os.pardir)
        os.chdir("photo")
        printed =[]
        # simulation()
        while streaming:
            d = {}
            data = open("classification_results.txt", "r", encoding='utf-8')
            os.chdir("photo_network")
            labels = data.readlines()
            for label in labels:
                pair = label.split(' ')
                # if (pair[1] not in faces_time):
                #     faces_time[pair[1]] = time.time()
                #     bot.send_photo(message.chat.id, open(pair[0], 'rb'))
                #     bot.send_message(message.chat.id, text="Обнаружен человек {}".format(pair[1]))
                    
                d[pair[0]] = pair[1]
                                               
            for filename in os.listdir():
                current_time = time.time()
                creation_time = os.path.getmtime(filename)
                mark = d[filename]
                if (filename.endswith(".jpg")) and (current_time - creation_time < 10) and (filename not in printed):
                    if (mark in faces_time):
                        if (current_time - faces_time[mark] >5):
                            bot.send_photo(message.chat.id, open(filename, 'rb'))
                            bot.send_message(message.chat.id, text="Обнаружен человек {}".format(mark))
                            faces_time[mark] = current_time
                            printed.append(filename)
                    else:
                        faces_time[mark] = current_time
                        bot.send_photo(message.chat.id, open(filename, 'rb'))
                        bot.send_message(message.chat.id, text="Обнаружен человек {}".format(mark))
                        printed.append(filename)
                    
                        
                #     else: 
                #         print("not ready {} {}".format(filename, time.time() - faces_time[d[filename]]))
                        
                # else:
                #     print("wrong format {}".format(filename))
    else:
        is_allowed(message)
        if (access):
            start(message)


def simulation():
    with open('classification_results.txt', 'w') as f:
            f.write('')
    k = 0
    cap = cv2.VideoCapture(0)
    for i in range(30):
        cap.read()
    while (k<=100):
        ret, frame = cap.read()
        cv2.imwrite('cam{}.jpg'.format(k), frame)
        os.chdir(os.pardir)
        with open('classification_results.txt', 'a') as f:
            f.write('cam{}.jpg'.format(k)+' {} \n'.format(k%2))
        k+=1
        os.chdir("photo_network")
    cap.release()


@bot.message_handler(commands=['check'])
def check(message):
    if (access):
        # Включаем первую камеру
        cap = cv2.VideoCapture(0)
        # "Прогреваем" камеру, чтобы снимок не был тёмным
        for i in range(30):
            cap.read()
        # Делаем снимок
        ret, frame = cap.read()
        # Записываем в файл
        if (ret): 
            cv2.imwrite(cam_dir + 'cam.png', frame)
            # Отключаем камеру
            bot.send_photo(message.chat.id, open(cam_dir + "cam.png", 'rb'))
        
        cap.release()
    else:
        is_allowed(message)
        if (access):
            check(message)

@bot.message_handler(commands=['learn'])
def learn(message):
    if (access):
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
        is_allowed(message)
        if (access):
            learn(message)

@bot.message_handler(commands=['improvement'])
def improve(message):
    if (access):
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

        # bot.send_photo(message.chat.id, open(animals_dir + "\\" + images[i], 'rb'))
        bot.send_message(message.chat.id, text="Это {}?".format())

    else:
        is_allowed(message)
        if (access):
            improve(message)

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
    if (access):
        if (streaming == True):
            bot.send_message(message.chat.id,
                             text="/learn - начать обучение\n/improvement - начать улучшение точности\n/report - получить отчет за последние сутки\n/check - получить кадр с камеры")

        elif (learning == True):
            bot.send_message(message.chat.id, text="/stop - закончить обучение\n/check - получить кадр с камеры")

        else:
            bot.send_message(message.chat.id,
                             text="/yes - подтвердить идентификацию\n/no - опровергнуть идентификацию\n/skip - пропустить некорректное изображение\n/stop - закончить улучшение точности\n/check - получить кадр с камеры")
    else:
        is_allowed(message)
        if (access):
            help(message)

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
        bot.send_message(message.chat.id, text="Доступ разрешен")
        global access
        access = True
        return True
    bot.send_message(message.chat.id, text="Доступ запрещен")
    return False

def get_names():
    with open("labels.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] not in names:
                names.append(row[1])

bot.polling(none_stop=True, interval=0)
