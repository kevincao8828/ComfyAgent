import os
import sys
import torch
import threading
import heapq
import logging
import copy
import time
from typing import List

import comfy.model_management
import execution
from execution import PromptQueue

class MultiGPUPromptQueue(PromptQueue):
    def __init__(self, server, gpu_ids):
        super().__init__(server)
        self.gpu_ids = gpu_ids
        self.gpu_queues = []
        self.gpu_threads = []
        
        # Create a queue for each GPU
        for gpu_id in gpu_ids:
            self.gpu_queues.append(PromptQueue(server))
            
        # Start worker threads for each GPU
        for i, gpu_id in enumerate(gpu_ids):
            thread = threading.Thread(
                target=prompt_worker, 
                daemon=True, 
                args=(self.gpu_queues[i], server, gpu_id)
            )
            thread.start()
            self.gpu_threads.append(thread)
            
        logging.info(f"Started {len(gpu_ids)} GPU worker threads")
    
    def put(self, item):
        with self.mutex:
            # Find the GPU with the smallest queue
            min_queue_size = float('inf')
            min_queue_idx = 0
            
            for i, gpu_queue in enumerate(self.gpu_queues):
                queue_size = gpu_queue.get_tasks_remaining()
                if queue_size < min_queue_size:
                    min_queue_size = queue_size
                    min_queue_idx = i
            
            # Add the task to the queue with the least tasks
            self.gpu_queues[min_queue_idx].put(item)
            heapq.heappush(self.queue, item)  # Keep track of all tasks for history
            self.server.queue_updated()
    
    def get_tasks_remaining(self):
        with self.mutex:
            total = 0
            for gpu_queue in self.gpu_queues:
                total += gpu_queue.get_tasks_remaining()
            return total
    
    def wipe_queue(self):
        with self.mutex:
            for gpu_queue in self.gpu_queues:
                gpu_queue.wipe_queue()
            self.queue = []
            self.server.queue_updated()
    
    def set_flag(self, name, data):
        with self.mutex:
            self.flags[name] = data
            for gpu_queue in self.gpu_queues:
                gpu_queue.set_flag(name, data)
            self.not_empty.notify()

def prompt_worker(q, server, gpu_id=None):
    if gpu_id is not None:
        # Set the GPU for this worker thread
        torch.cuda.set_device(gpu_id)
        logging.info(f"Worker thread using GPU {gpu_id} (CUDA device {torch.cuda.current_device()})")
    
    e = execution.PromptExecutor(server)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            current_time = time.perf_counter()
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server.last_prompt_id = prompt_id

            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id,
                        e.outputs_ui,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))
            if server.client_id is not None:
                server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, server.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                comfy.model_management.cleanup_models()
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False