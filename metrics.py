
from datetime import datetime
import time

import GPUtil
import psutil


class Metrics:

    def __init__(self):
        self.log_intialized = False
        self.active_ias = ""
        self.capture_objects = False

    def reinitialize(self):
        self.log_intialized = False
    
    def log_performance(self):
        
        if not self.log_intialized:
            self.log_name = f"performance_log_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"

            with open(self.log_name, "a") as log_file:
                log_file.write(f"Active AIs: {self.active_ias}\n")
                log_file.write(f"Capture objects: {self.capture_objects}\n")
                log_file.write("Time, CPU_Usage (%), RAM_Usage (%), GPU_usage (%), GPU_memory_usage, cycle_time (s)\n")
                log_file.flush()
            
            self.log_intialized = True
            self.previous_time = time.time()
        
        else:

            
            current_time = time.time()
            timestamp = datetime.now().strftime("%H:%M:%S")
            cycle_time = current_time - self.previous_time
            self.previous_time = current_time

            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent

            # Captura métricas da GPU
            gpu_usage = "N/A"
            gpu_memory_usage = "N/A"

            try:
                gpus = GPUtil.getGPUs()  # Obtém informações das GPUs
                if gpus:
                    gpu_usage = f"{(gpus[0].load * 100):.2f}%"  # Percentual de uso da GPU (primeira GPU)
                    gpu_memory_usage = f"{(gpus[0].memoryUtil * 100):.2f}%"  # Percentual de memória usada
            except Exception as e:
                None

            with open(self.log_name, "a") as log_file:
                log_file.write(f"{timestamp}, {cpu_usage}%, {ram_usage}%, {gpu_usage}, {gpu_memory_usage}, {cycle_time:.2f}s\n")
                log_file.flush()