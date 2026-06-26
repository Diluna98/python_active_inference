import multiprocessing
import threading
import time
import psutil


def cpu_availability():
    usage = psutil.cpu_percent(interval=1)
    return max(0.0, 100.0 - usage)


def partial_load(duty_cycle=0.1):

    busy_time = duty_cycle
    idle_time = 1.0 - duty_cycle

    while True:

        start = time.perf_counter()

        while (time.perf_counter() - start) < busy_time:
            pass

        time.sleep(idle_time)


if __name__ == "__main__":

    processes = []

    # create workers ONCE
    for _ in range(4): #make this 20 to simulate about 50-60% load
                        #make this 4 to simulate about 0-10% load 
                        #make this 30 to simulate about 70-80% load

        p = multiprocessing.Process(
            target=partial_load,
            args=(0.9,)
        )

        p.start()
        processes.append(p)