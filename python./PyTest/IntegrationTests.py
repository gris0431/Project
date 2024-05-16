import unittest
import json
import sqlite3
import cv2
import telebot
import pika
from TgBot import bot
import numpy as np
from broker import Broker
from telegram_bot import TelegramBot
from unittest.mock import Mock, patch
from neural_network_module import NeuralNetworkModule
from neural_network_module import NeuralNetworkModule
from image_processing_module import ImageProcessingModule
from module_retraining import ModuleRetraining
from module_neural_network import ModuleNeuralNetwork
from control_module import ControlModule
from telegram_server import TelegramServer
from database_module import DatabaseModule
from database import Database
from neural_network_retraining_module import NeuralNetworkRetrainingModule

#1 Тест взаимодействия Телеграм-интерфейса с Телеграм-сервером
class TestTelegramInteraction(unittest.TestCase):

    def setUp(self):
        self.bot = telebot.TeleBot('TOKEN')

    @patch('telebot.TeleBot.send_message')
    def test_send_command(self, mock_send_message):
        # Шаг 1: Отправить команду обучения или уведомления через Телеграм
        self.bot.process_new_message(telebot.types.Message(text='/start', chat=telebot.types.Chat(id=12345)))

        # Шаг 2: Проверить, что Телеграм-сервер получил соответствующий запрос и обработал его без ошибок
        mock_send_message.assert_called_once_with(chat_id=12345, text='Режим дежурства')

        # Шаг 3: Проверить логирование и статус запроса на сервере
        # Здесь можно добавить проверки логов и статуса запроса на сервере

#2 Тест взаимодействия Телеграм-сервера с Телеграм-ботом        
class TestTelegramServerInteraction(unittest.TestCase):

    def setUp(self):
        self.bot = telebot.TeleBot('TOKEN')

    @patch('telebot.TeleBot.send_message')
    @patch('telebot.TeleBot.send_photo')
    def test_receive_request(self, mock_send_photo, mock_send_message):
        # Шаг 1: Имитировать запрос на обучение или уведомление на Телеграм-сервере
        update = Mock()
        update.message = Mock()
        update.message.chat = Mock()
        update.message.chat.id = 12345
        update.message.text = '/train'

        # Шаг 2: Проверить, что Телеграм-бот получает запрос и выполняет соответствующие действия
        self.bot.process_new_updates([update])

        mock_send_message.assert_called_once_with(chat_id=12345, text='Режим обучения')
        mock_send_photo.assert_called_once_with(chat_id=12345, photo=open('AnimalTest/animals_0.png', 'rb'))

#3 Тест взаимодействия Телеграм-бота с Брокером сообщений        
class TestTelegramBotInteractionWithMessageBroker(unittest.TestCase):

    @patch('pika.BlockingConnection')
    def test_send_command(self, mock_connection):
        # Создаем мок-объект для соединения с брокером сообщений
        connection = mock_connection.return_value
        channel = Mock()
        connection.channel.return_value = channel

        # Создаем Телеграм-бота
        bot = Mock()
        bot.send_message = Mock()

        # Создаем обработчик команды для Телеграм-бота
        def handle_command(update, context):
            # Отправляем сообщение в брокер сообщений
            with connection:
                channel.basic_publish(
                    exchange='test_exchange',
                    routing_key='test_routing_key',
                    body=json.dumps({'command': 'test_command', 'data': 'test_data'})
                )

        # Добавляем обработчик команды в Телеграм-бота
        bot.command('test', handle_command)

        # Отправляем команду через Телеграм-бота
        update = Mock()
        context = Mock()
        handle_command(update, context)

        # Проверяем, что брокер сообщений получил сообщение с правильными данными
        channel.basic_publish.assert_called_once_with(
            exchange='test_exchange',
            routing_key='test_routing_key',
            body=json.dumps({'command': 'test_command', 'data': 'test_data'})
        )

        # Проверяем, что Телеграм-бот отправил ответное сообщение
        bot.send_message.assert_called_once_with(chat_id=12345, text='Команда отправлена')
        
#4 Тест взаимодействия Брокера сообщений с Модулем нейросети
class TestNeuralNetworkModuleInteraction(unittest.TestCase):

    def setUp(self):
        self.broker = Broker()
        self.neural_network = NeuralNetworkModule()

    def test_image_processing(self):
        # Шаг 1: Отправить изображение на обработку через Телеграм-бота
        image = cv2.imread('test_image.jpg')
        message = {'image': image}

        # Шаг 2: Проверить, что брокер сообщений передает изображение в модуль нейросети
        self.broker.send_message('neural_network', message)
        received_message = self.broker.get_message('neural_network')
        self.assertEqual(received_message, message)

        # Шаг 3: Убедиться, что модуль нейросети возвращает результаты кластеризации через брокер сообщений
        result = self.neural_network.process_image(image)
        self.broker.send_message('telegram_bot', {'result': result})
        received_result = self.broker.get_message('telegram_bot')
        self.assertEqual(received_result, {'result': result})
        
#5 Тест взаимодействия Модуля нейросети и Базы данных        
class TestNeuralNetworkDatabaseInteraction(unittest.TestCase):

    def setUp(self):
        self.db = Database('test.db')
        self.nn = NeuralNetworkModule()

    def tearDown(self):
        self.db.close_connection()
        os.remove('test.db')

    def test_save_and_load_data(self):
        # Шаг 1: Отправить изображение на обработку и кластеризацию
        image = cv2.imread('test_image.jpg')
        faces, clusters = self.nn.process_image(image)

        # Шаг 2: Проверить, что результаты распознавания лиц и кластеров сохраняются в базе данных
        self.db.save_faces(faces)
        self.db.save_clusters(clusters)

        saved_faces = self.db.load_faces()
        saved_clusters = self.db.load_clusters()

        np.testing.assert_array_equal(faces, saved_faces)
        np.testing.assert_array_equal(clusters, saved_clusters)

        # Шаг 3: Убедиться, что данные могут быть успешно извлечены из базы данных
        loaded_faces = self.db.load_faces()
        loaded_clusters = self.db.load_clusters()

        np.testing.assert_array_equal(faces, loaded_faces)
        np.testing.assert_array_equal(clusters, loaded_clusters)

#6 Тест взаимодействия Модуля обработки изображений с Базой данных и Модулем нейросети
class TestImageProcessingInteraction(unittest.TestCase):

    def setUp(self):
        self.telegram_bot = TelegramBot(token='test_token')
        self.image_processing_module = ImageProcessingModule()
        self.neural_network_module = NeuralNetworkModule()
        self.database = Database(db_name='test_db')

    def tearDown(self):
        self.database.delete_all_images()

    def test_image_processing_interaction(self):
        # Шаг 1: Загрузить изображение через Телеграм-бота
        image = cv2.imread('test_image.jpg')
        message = Mock()
        message.photo = [-1, {'file_id': 'test_file_id'}]
        self.telegram_bot.send_message = Mock()
        self.telegram_bot.get_file = Mock(return_value={'file_path': 'test_file_path'})
        self.telegram_bot.download_file = Mock(return_value=image)

        # Шаг 2: Убедиться, что модуль обработки изображений корректно обрабатывает изображение
        processed_image = self.image_processing_module.process_image(image)
        self.assertIsNotNone(processed_image)

        # Шаг 3: Проверить передачу обработанного изображения в базу данных и модуль нейросети
        self.database.save_image(processed_image)
        saved_image = self.database.get_image(1)
        self.assertIsNotNone(saved_image)
        np.testing.assert_array_equal(processed_image, saved_image)

        self.neural_network_module.process_image(processed_image)
        self.neural_network_module.get_result.assert_called_once()
        
#7 Тест взаимодействия Управляющего модуля с остальными модулями       
class TestControlModuleInteraction(unittest.TestCase):

    def setUp(self):
        self.image_processing_module = ImageProcessingModule()
        self.neural_network_module = NeuralNetworkModule()
        self.database_module = DatabaseModule()
        self.control_module = ControlModule(self.image_processing_module, self.neural_network_module, self.database_module)

    def test_reclustering(self):
        # Шаг 1: Имитировать сценарий, требующий взаимодействия нескольких модулей (например, пересборка кластеров при неверном распознавании)
        image = Mock()
        self.image_processing_module.process_image = Mock(return_value=image)
        self.neural_network_module.recognize_faces = Mock(return_value=[(1, 0.5), (2, 0.4)])
        self.database_module.get_clusters = Mock(return_value=[(1, [1, 2, 3]), (2, [4, 5, 6])])
        self.database_module.update_clusters = Mock()

        # Шаг 2: Проверить координацию управляющего модуля в процессе выполнения задач
        self.control_module.recluster_faces()

        # Проверка взаимодействия с модулем обработки изображений
        self.image_processing_module.process_image.assert_called_once()

        # Проверка взаимодействия с модулем нейросети
        self.neural_network_module.recognize_faces.assert_called_once_with(image)

        # Проверка взаимодействия с модулем базы данных
        self.database_module.get_clusters.assert_called_once()
        self.database_module.update_clusters.assert_called_once()
        
#8 Тест взаимодействия Модуля переобучения нейросети с Модулем нейросети и Базой данных        
class TestModuleRetrainingInteraction(unittest.TestCase):

    def setUp(self):
        self.mock_db = Mock(spec=Database)
        self.mock_nn = Mock(spec=ModuleNeuralNetwork)
        self.retraining_module = ModuleRetraining(self.mock_db, self.mock_nn)

    def test_retrain_neural_network(self):
        # Шаг 1: Вызвать процесс переобучения нейросети
        self.retraining_module.retrain_neural_network()

        # Шаг 2: Проверить, что модуль переобучения корректно взаимодействует с базой данных и модулем нейросети
        self.mock_db.get_all_images.assert_called_once()
        self.mock_nn.retrain.assert_called_once()

        # Шаг 3: Убедиться, что пересобранные кластеры обновлены и хранятся в базе данных
        self.mock_db.update_clusters.assert_called_once()
        
#9 Взаимодействие управляющего модуля с Телеграм-сервером       
class TestControlModuleInteraction(unittest.TestCase):

    def setUp(self):
        self.control_module = ControlModule()
        self.telegram_server = TelegramServer()

    def test_handle_request(self):
        # Создаем мок-объект запроса
        request_data = {
            "update_id": 12345678,
            "message": {
                "message_id": 7890,
                "from": {
                    "id": 1234567,
                    "is_bot": False,
                    "first_name": "John",
                    "last_name": "Doe",
                    "username": "johndoe"
                },
                "chat": {
                    "id": 1234567,
                    "first_name": "John",
                    "last_name": "Doe",
                    "username": "johndoe",
                    "type": "private"
                },
                "date": 1617084342,
                "text": "/train"
            }
        }
        request_json = json.dumps(request_data)

        # Создаем мок-объект для функции обработки запросов модуля нейросети
        neural_network_mock = Mock()
        neural_network_mock.handle_request.return_value = "Training started"

        # Заменяем модуль нейросети на мок-объект
        self.control_module.neural_network_module = neural_network_mock

        # Вызываем функцию обработки запросов управляющего модуля
        response = self.control_module.handle_request(request_json)

        # Проверяем, что функция обработки запросов модуля нейросети была вызвана с правильными аргументами
        neural_network_mock.handle_request.assert_called_once_with(request_data)

        # Проверяем, что функция возвращает правильный ответ
        expected_response = {
            "chat_id": 1234567,
            "text": "Training started"
        }
        self.assertEqual(response, expected_response)
        
#10 Взаимодействие управляющего модуля с Телеграм-ботом       
class TestControlModuleInteraction(unittest.TestCase):
    def setUp(self):
        self.control_module = ControlModule()
        self.bot = TelegramBot()

    @patch('telegram.Bot.send_message')
    def test_handle_training_request(self, mock_send_message):
        # Имитируем запрос на обучение от бота
        request = {'text': '/train'}
        self.bot.last_message = Mock(text=request['text'])

        # Вызываем метод handle_request управляющего модуля
        self.control_module.handle_request(self.bot)

        # Проверяем, что метод send_message был вызван с правильными аргументами
        mock_send_message.assert_called_once_with(chat_id=self.bot.chat_id, text='Режим обучения')

    @patch('telegram.Bot.send_photo')
    def test_handle_photo_request(self, mock_send_photo):
        # Имитируем запрос на уточнение фотографии от бота
        request = {'text': '/photo', 'photo': ['file_id']}
        self.bot.last_message = Mock(text=request['text'], photo=request['photo'])

        # Вызываем метод handle_request управляющего модуля
        self.control_module.handle_request(self.bot)

        # Проверяем, что метод send_photo был вызван с правильными аргументами
        mock_send_photo.assert_called_once_with(chat_id=self.bot.chat_id, photo=request['photo'][0])

    @patch('telegram.Bot.send_message')
    def test_handle_notification(self, mock_send_message):
        # Имитируем уведомление от бота
        notification = {'text': 'Новое лицо обнаружено'}
        self.bot.last_message = Mock(text=notification['text'])

        # Вызываем метод handle_request управляющего модуля
        self.control_module.handle_request(self.bot)

        # Проверяем, что метод send_message был вызван с правильными аргументами
        mock_send_message.assert_called_once_with(chat_id=self.bot.chat_id, text=notification['text'])
        
#11 Взаимодействие управляющего модуля с брокером сообщений        
class TestControlModuleInteraction(unittest.TestCase):

    def setUp(self):
        self.control_module = Mock()
        self.rabbitmq_connection = pika.BlockingConnection(
            pika.ConnectionParameters('rabbitmq'))
        self.channel = self.rabbitmq_connection.channel()
        self.queue_name = 'control_queue'
        self.channel.queue_declare(queue=self.queue_name)

    def tearDown(self):
        self.channel.close()
        self.rabbitmq_connection.close()

    def test_send_command(self):
        command = {'command': 'train', 'data': {'epochs': 10}}
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=json.dumps(command))

        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.channel.start_consuming()
        self.control_module.handle_response.assert_called_once_with(command['command'], command['data'])

    def on_response(self, ch, method, props, body):
        response = json.loads(body)
        self.control_module.handle_response(response['command'], response['data'])
        
#12 Взаимодействие управляющего модуля с базой данных
class TestControlModuleDatabaseInteraction(unittest.TestCase):

    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.control_module = ControlModule(db=self.db)

    def tearDown(self):
        self.db.close()

    def test_save_recognition_results(self):
        recognition_results = [
            {'id': 1, 'name': 'John', 'confidence': 0.9},
            {'id': 2, 'name': 'Alice', 'confidence': 0.85},
        ]
        self.control_module.save_recognition_results(recognition_results)

        cursor = self.db.cursor()
        cursor.execute('SELECT * FROM recognition_results')
        results = cursor.fetchall()

        expected_results = [
            (1, 'John', 0.9),
            (2, 'Alice', 0.85),
        ]
        self.assertEqual(results, expected_results)

    def test_get_recognition_results(self):
        recognition_results = [
            {'id': 1, 'name': 'John', 'confidence': 0.9},
            {'id': 2, 'name': 'Alice', 'confidence': 0.85},
        ]
        cursor = self.db.cursor()
        cursor.executemany('''
            INSERT INTO recognition_results (id, name, confidence) VALUES (?, ?, ?)
        ''', recognition_results)
        self.db.commit()

        results = self.control_module.get_recognition_results()

        expected_results = [
            {'id': 1, 'name': 'John', 'confidence': 0.9},
            {'id': 2, 'name': 'Alice', 'confidence': 0.85},
        ]
        self.assertEqual(results, expected_results)        

#13 Взаимодействие управляющего модуля с модулем нейросети
class TestControlModuleNeuralNetworkInteraction(unittest.TestCase):
    def setUp(self):
        self.control_module = ControlModule()
        self.neural_network_module = NeuralNetworkModule()

    @patch('neural_network_module.NeuralNetworkModule.recognize_faces')
    @patch('neural_network_module.NeuralNetworkModule.cluster_images')
    def test_process_image(self, mock_cluster_images, mock_recognize_faces):
        # Создаем мок-объекты для результатов распознавания и кластеризации
        recognize_result = [{'name': 'John', 'confidence': 0.9}, {'name': 'Jane', 'confidence': 0.8}]
        cluster_result = [{'cluster_id': 1, 'image_paths': ['image1.jpg', 'image2.jpg']}]

        # Задаем возвращаемые значения для мок-методов
        mock_recognize_faces.return_value = recognize_result
        mock_cluster_images.return_value = cluster_result

        # Вызываем метод process_image управляющего модуля
        result = self.control_module.process_image('image.jpg')

        # Проверяем, что методы recognize_faces и cluster_images были вызваны с правильными аргументами
        mock_recognize_faces.assert_called_once_with('image.jpg')
        mock_cluster_images.assert_called_once_with(['image.jpg'])

        # Проверяем, что результаты распознавания и кластеризации были переданы в другие модули
        self.neural_network_module.save_recognition_results.assert_called_once_with(recognize_result)
        self.neural_network_module.save_clustering_results.assert_called_once_with(cluster_result)

        # Проверяем, что метод process_image вернул правильный результат
        expected_result = {'recognition_results': recognize_result, 'clustering_results': cluster_result}
        self.assertEqual(result, expected_result)
        
#14 Взаимодействие управляющего модуля с модулем обработки изображений
class TestControlModuleImageProcessingInteraction(unittest.TestCase):
    def setUp(self):
        self.control_module = ControlModule()
        self.image_processing_module = ImageProcessingModule()
        self.image_processing_module.process_image = Mock()

    def test_process_image(self):
        # Отправляем изображение на обработку
        image = 'test_image.jpg'
        self.control_module.process_image(image)

        # Проверяем, что модуль обработки изображений получил изображение на обработку
        self.image_processing_module.process_image.assert_called_once_with(image)

    def test_process_and_save_image(self):
        # Отправляем изображение на обработку и сохранение в базу данных
        image = 'test_image.jpg'
        processed_image = 'processed_image.jpg'
        self.image_processing_module.process_image.return_value = processed_image
        self.control_module.save_image = Mock()

        self.control_module.process_and_save_image(image)

        # Проверяем, что модуль обработки изображений получил изображение на обработку
        self.image_processing_module.process_image.assert_called_once_with(image)
        # Проверяем, что обработанное изображение было сохранено в базу данных
        self.control_module.save_image.assert_called_once_with(processed_image)
        
#15 Взаимодействие управляющего модуля с модулем переобучения нейросети
class TestControlModuleNeuralNetworkRetrainingInteraction(unittest.TestCase):
    def setUp(self):
        self.control_module = ControlModule()
        self.retraining_module = NeuralNetworkRetrainingModule()
        self.retraining_module.retrain_neural_network = Mock()

    def test_initiate_retraining(self):
        # Управляющий модуль инициирует процесс переобучения нейросети
        self.control_module.initiate_retraining()

        # Проверяем, что модуль переобучения нейросети получил команду на переобучение
        self.retraining_module.retrain_neural_network.assert_called_once()

    def test_retrain_and_save_results(self):
        # Управляющий модуль инициирует процесс переобучения нейросети и получает обновленные результаты
        retraining_results = {'new_clusters': [1, 2, 3], 'new_weights': [0.1, 0.2, 0.3]}
        self.retraining_module.retrain_neural_network.return_value = retraining_results
        self.control_module.save_clusters = Mock()
        self.control_module.update_neural_network = Mock()

        self.control_module.retrain_and_save_results()

        # Проверяем, что модуль переобучения нейросети получил команду на переобучение
        self.retraining_module.retrain_neural_network.assert_called_once()
        # Проверяем, что обновленные кластеры были сохранены в базу данных
        self.control_module.save_clusters.assert_called_once_with(retraining_results['new_clusters'])
        # Проверяем, что обновленные веса были переданы в модуль нейросети
        self.control_module.update_neural_network.assert_called_once_with(retraining_results['new_weights'])
        
#16 Взаимодействие управляющего модуля с остальными подсистемами
class TestControlModuleInteraction(unittest.TestCase):
    def setUp(self):
        self.control_module = ControlModule()
        self.control_module.image_processing_module = Mock()
        self.control_module.neural_network_module = Mock()
        self.control_module.database_module = Mock()
        self.control_module.retraining_module = Mock()

    def test_process_image(self):
        # Отправляем изображение на обработку
        image_path = 'test_image.jpg'
        self.control_module.process_image(image_path)

        # Проверяем, что модуль обработки изображений получил изображение на обработку
        self.control_module.image_processing_module.process_image.assert_called_once_with(image_path)
        # Проверяем, что модуль нейросети получил обработанное изображение для распознавания
        self.control_module.neural_network_module.recognize_image.assert_called_once()
        # Проверяем, что результаты распознавания были сохранены в базу данных
        self.control_module.database_module.save_recognition_result.assert_called_once()

    def test_retrain_neural_network(self):
        # Инициируем процесс переобучения нейросети
        self.control_module.retrain_neural_network()

        # Проверяем, что модуль переобучения нейросети получил команду на переобучение
        self.control_module.retraining_module.retrain.assert_called_once()
        # Проверяем, что обновленная нейросеть была сохранена в базу данных
        self.control_module.database_module.save_neural_network.assert_called_once()

    def test_system_status_check(self):
        # Имитируем сбой в работе модуля обработки изображений
        self.control_module.image_processing_module.is_available.return_value = False

        # Вызываем метод проверки состояния системы
        self.control_module.check_system_status()

        # Проверяем, что управляющий модуль выполнил коррекцию работы модуля обработки изображений
        self.control_module.image_processing_module.recover.assert_called_once()
                
if __name__ == '__main__':
    unittest.main()
