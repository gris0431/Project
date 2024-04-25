import concurrent.futures
import subprocess
import time

def run_comb_script():
    try:
        subprocess.run(["python", "comb.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running comb.py: {e}")

def run_neural_network_script():
    try:
        while True:
            subprocess.run(["python", "neuralnetwork.py"], check=True)
            time.sleep(5)  # Wait for 5 seconds before running again
    except subprocess.CalledProcessError as e:
        print(f"Error running neuralnetwork.py: {e}")

# Run both scripts in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_comb = executor.submit(run_comb_script)
    future_neural_network = executor.submit(run_neural_network_script)

    # Wait for both processes to finish
    concurrent.futures.wait([future_comb])

print("Both scripts have finished executing")