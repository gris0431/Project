import concurrent.futures
import subprocess
import time

def run_comb_script():
    # Запуск скрипта comb.py
    try:
        subprocess.run(["python", "comb.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running comb.py: {e}")

def run_neural_network_script():
    # Запуск скрипта neuralnetwork.py с периодичностью
    try:
        while True:
            subprocess.run(["python", "neuralnetwork.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running neuralnetwork.py: {e}")

# Запуск обоих скриптов параллельно
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_comb = executor.submit(run_comb_script)
    future_neural_network = executor.submit(run_neural_network_script)

    # Ожидание завершения обоих процессов
    concurrent.futures.wait([future_comb])

    # Если future_comb завершен, отменить future_neural_network
    future_neural_network.cancel()

print("Both scripts have finished executing")
