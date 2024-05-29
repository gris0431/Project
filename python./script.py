import concurrent.futures
import subprocess
import time
import threading

def run_comb_script():
    try:
        subprocess.run(["python", "Observer.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running comb.py: {e}")

def run_neural_network_script(stop_event):
    try:
        while not stop_event.is_set():
            subprocess.run(["python", "neuralnetwork.py"], check=True)
            time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running neuralnetwork.py: {e}")

def run_main_script(stop_event):
    try:
        while not stop_event.is_set():
            subprocess.run(["python", "neu2.py"], check=True)
            time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running neu2.py: {e}")

def run_bot_script(stop_event):
    try:
        while not stop_event.is_set():
            subprocess.run(["python", "bot.py"], check=True)
            time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running bot.py: {e}")

stop_event = threading.Event()

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_comb = executor.submit(run_comb_script)
    future_neural_network = executor.submit(run_neural_network_script, stop_event)
    future_main = executor.submit(run_main_script, stop_event)
    future_bot = executor.submit(run_bot_script, stop_event)

    concurrent.futures.wait([future_comb])

    stop_event.set()

print("comb.py has finished executing, other scripts have been stopped")
