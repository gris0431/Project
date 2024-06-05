import concurrent.futures
import subprocess
import time
import threading
import sys
import os

def run_comb_script(python_executable):
    try:
        subprocess.run([python_executable, "Observer.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running comb.py: {e}")

def run_neural_network_script(stop_event, python_executable):
    try:
        while not stop_event.is_set():
            subprocess.run([python_executable, "neuralnetwork.py"], check=True)
            time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running neuralnetwork.py: {e}")

def run_main_script(stop_event, python_executable):
    try:
        while not stop_event.is_set():
            subprocess.run([python_executable, "neu2.py"], check=True)
            time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running neu2.py: {e}")

def run_bot_script(stop_event, python_executable):
    try:
        while not stop_event.is_set():
            subprocess.run([python_executable, "bot.py"], check=True)
            time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running bot.py: {e}")

stop_event = threading.Event()
python_executable = sys.executable

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_comb = executor.submit(run_comb_script, python_executable)
    future_neural_network = executor.submit(run_neural_network_script, stop_event, python_executable)
    future_main = executor.submit(run_main_script, stop_event, python_executable)
    future_bot = executor.submit(run_bot_script, stop_event, python_executable)

    concurrent.futures.wait([future_comb])

    stop_event.set()

print("comb.py has finished executing, other scripts have been stopped")
