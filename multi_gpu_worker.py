import os
import sys
import time
import torch
import logging
import gc
import multiprocessing as mp
from multiprocessing import Process, Queue

import execution
import comfy.model_management

class GPUWorkerProcess(Process):
    """Worker process that handles task execution on a specific GPU"""
    
    def __init__(self, gpu_id, task_queue, result_queue, server):
        super().__init__()
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.server = server
        self.daemon = True
        
    def run(self):
        """Main worker process loop"""
        # Set the CUDA_VISIBLE_DEVICES for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        logging.info(f"GPU Worker {self.gpu_id} started")
        
        # Initialize executor for this worker
        executor = execution.PromptExecutor(self.server)
        last_gc_collect = 0
        need_gc = False
        gc_collect_interval = 10.0
        
        while True:
            try:
                # Get task with timeout
                timeout = 1000.0
                if need_gc:
                    timeout = max(gc_collect_interval - (time.perf_counter() - last_gc_collect), 0.0)
                
                try:
                    queue_item = self.task_queue.get(timeout=timeout)
                except mp.queues.Empty:
                    queue_item = None
                
                if queue_item is not None:
                    item, item_id = queue_item
                    execution_start_time = time.perf_counter()
                    prompt_id = item[1]
                    
                    # Send status update that this GPU is processing the task
                    self.result_queue.put({
                        'type': 'status_update',
                        'status': 'executing',
                        'gpu_id': self.gpu_id,
                        'prompt_id': prompt_id,
                        'item_id': item_id
                    })
                    
                    # Execute the prompt
                    executor.execute(item[2], prompt_id, item[3], item[4])
                    need_gc = True
                    
                    # Send completion status and results
                    self.result_queue.put({
                        'type': 'task_done',
                        'item_id': item_id,
                        'outputs': executor.outputs_ui,
                        'status': execution.PromptQueue.ExecutionStatus(
                            status_str='success' if executor.success else 'error',
                            completed=executor.success,
                            messages=executor.status_messages
                        ),
                        'gpu_id': self.gpu_id,
                        'prompt_id': prompt_id
                    })
                    
                    current_time = time.perf_counter()
                    execution_time = current_time - execution_start_time
                    logging.info(f"GPU {self.gpu_id}: Prompt executed in {execution_time:.2f} seconds")
                
                # Check if we need to perform garbage collection
                if need_gc:
                    current_time = time.perf_counter()
                    if (current_time - last_gc_collect) > gc_collect_interval:
                        comfy.model_management.cleanup_models()
                        gc.collect()
                        comfy.model_management.soft_empty_cache()
                        last_gc_collect = current_time
                        need_gc = False
                
                # Check for special commands
                try:
                    command = self.task_queue.get(block=False)
                    if command == "EXIT":
                        break
                    elif isinstance(command, dict):
                        if command.get('type') == 'unload_models':
                            comfy.model_management.unload_all_models()
                            need_gc = True
                            last_gc_collect = 0
                        elif command.get('type') == 'free_memory':
                            executor.reset()
                            need_gc = True
                            last_gc_collect = 0
                except mp.queues.Empty:
                    pass
                    
            except Exception as e:
                logging.error(f"GPU Worker {self.gpu_id} error: {e}")
                logging.error(f"Traceback: {sys.exc_info()[2]}")
                # Try to recover and continue
                try:
                    executor.reset()
                    comfy.model_management.cleanup_models()
                    gc.collect()
                    comfy.model_management.soft_empty_cache()
                except:
                    pass
        
        logging.info(f"GPU Worker {self.gpu_id} shutting down")


class MultiGPUManager:
    """Manages multiple GPU worker processes and distributes tasks"""
    
    def __init__(self, gpu_ids, server, prompt_queue):
        self.gpu_ids = gpu_ids
        self.server = server
        self.prompt_queue = prompt_queue
        self.workers = []
        self.task_queues = []
        self.result_queue = Queue()
        self.gpu_status = {gpu_id: {'busy': False, 'tasks_completed': 0, 'current_task': None} for gpu_id in gpu_ids}
        
        # Create task queues and worker processes for each GPU
        for gpu_id in gpu_ids:
            task_queue = Queue()
            self.task_queues.append(task_queue)
            worker = GPUWorkerProcess(gpu_id, task_queue, self.result_queue, server)
            self.workers.append(worker)
        
        # Start all worker processes
        for worker in self.workers:
            worker.start()
        
        logging.info(f"Multi-GPU manager initialized with GPUs: {gpu_ids}")
    
    def process_results(self):
        """Process results from worker processes"""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get(block=False)
                
                if result['type'] == 'status_update':
                    gpu_id = result['gpu_id']
                    self.gpu_status[gpu_id]['busy'] = True
                    self.gpu_status[gpu_id]['current_task'] = result['prompt_id']
                    
                elif result['type'] == 'task_done':
                    item_id = result['item_id']
                    outputs = result['outputs']
                    status = result['status']
                    gpu_id = result['gpu_id']
                    
                    # Mark the GPU as free
                    self.gpu_status[gpu_id]['busy'] = False
                    self.gpu_status[gpu_id]['tasks_completed'] += 1
                    self.gpu_status[gpu_id]['current_task'] = None
                    
                    # Update the prompt queue with the results
                    self.prompt_queue.task_done(item_id, outputs, status)
        except Exception as e:
            logging.error(f"Error processing results: {e}")
    
    def get_next_available_gpu(self):
        """Simple load balancing - return the least busy GPU"""
        available_gpus = [gpu_id for gpu_id in self.gpu_ids if not self.gpu_status[gpu_id]['busy']]
        
        if not available_gpus:
            # All GPUs are busy, return None
            return None
        
        # Return the GPU with the least completed tasks (simple load balancing)
        return min(available_gpus, key=lambda gpu_id: self.gpu_status[gpu_id]['tasks_completed'])
    
    def process_queue(self):
        """Process items from the prompt queue and distribute to workers"""
        # First process any results from workers
        self.process_results()
        
        # Check if there's an available GPU
        next_gpu = self.get_next_available_gpu()
        if next_gpu is None:
            return
        
        # Get the next item from the queue
        queue_item = self.prompt_queue.get_next_item()
        if queue_item is None:
            return
        
        # Get the index of the GPU in our list
        gpu_index = self.gpu_ids.index(next_gpu)
        
        # Send the task to the appropriate worker
        self.task_queues[gpu_index].put(queue_item)
        
        # Update GPU status
        self.gpu_status[next_gpu]['busy'] = True
    
    def send_command_to_all_workers(self, command):
        """Send a command to all worker processes"""
        for task_queue in self.task_queues:
            task_queue.put(command)
    
    def shutdown(self):
        """Shutdown all worker processes"""
        for task_queue in self.task_queues:
            task_queue.put("EXIT")
        
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        
        logging.info("All GPU workers shut down")