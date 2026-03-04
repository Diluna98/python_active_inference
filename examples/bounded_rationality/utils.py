import multiprocessing
import numpy as np
import psutil
import threading
import time

def latency_bin_index(average_latency: float) -> int:
    """
    Returns the bin index (starting from 0) for the given average_latency.

    Bins:
    0: <= 1
    1: (1, 1.7]
    2: (1.7, 2.4]
    3: (2.4, 3.1]
    4: (3.1, 3.8]
    5: (3.8, 4.5]
    6: (4.5, 5.2]
    7: (5.2, 5.9]
    8: (5.9, 6.6]
    9: > 6.6
    """
    bounds = [1.0, 1.7, 2.4, 3.1, 3.8, 4.5, 5.2, 5.9, 6.6]

    for i, upper in enumerate(bounds):
        if average_latency <= upper:
            return i

    return len(bounds)

def model_evidence_bin_index(model_evidence: float) -> int:
    """
    Returns the bin index starting from 0 for the given model_evidence.

    Bins:
    0: <= -0.009
    1: (-0.009, -0.008]
    2: (-0.008, -0.007]
    3: (-0.007, -0.006]
    4: (-0.006, -0.005]
    5: (-0.005, -0.004]
    6: (-0.004, -0.003]
    7: (-0.003, -0.002]
    8: (-0.002, -0.001]
    9: > -0.001
    """
    bounds = [-0.009, -0.008, -0.007, -0.006, -0.005,
              -0.004, -0.003, -0.002, -0.001]

    for i, upper in enumerate(bounds):
        if model_evidence <= upper:
            return i

    return len(bounds)



def burn_cpu_matrix(size=20):
    # continuously multiply large matrices
    while True:
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)  # heavy CPU work

def monitor_cpu(cpu_list, interval=0.01, stop_flag=[True]):
    """Continuously record CPU usage into cpu_list until stop_flag[0] is False."""
    psutil.cpu_percent(None)  # prime
    while stop_flag[0]:
        cpu_list.append(psutil.cpu_percent(interval=None))
        time.sleep(interval)

if __name__ == "__main__":
    workers = []
    num_workers = 4  # number of cores to stress
    matrix_size = 10  # safe size

    # start CPU load workers
    for _ in range(num_workers):
        p = multiprocessing.Process(target=burn_cpu_matrix, args=(matrix_size,))
        p.start()
        workers.append(p)

    # start CPU monitoring thread
    cpu_usages = []
    stop_flag = [True]
    monitor_thread = threading.Thread(target=monitor_cpu, args=(cpu_usages, 0.01, stop_flag))
    monitor_thread.start()

    # run the stress test for a fixed duration
    test_duration = 30  # seconds
    time.sleep(test_duration)

    # stop monitoring
    stop_flag[0] = False
    monitor_thread.join()

    # stop CPU load workers
    for p in workers:
        p.terminate()
        p.join()

    # compute mean CPU usage
    mean_cpu = sum(cpu_usages) / len(cpu_usages)
    print(f"Mean CPU usage during test: {mean_cpu:.2f}%")