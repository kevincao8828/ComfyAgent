import os
import sys
import time
import torch
import logging
import threading
import multiprocessing
from multiprocessing import shared_memory
import queue
import json
import numpy as np
import pickle
import traceback

# Dictionary to track shared models across processes
shared_models = {}
# Lock for accessing shared models
shared_models_lock = multiprocessing.Lock()

class GPUWorkerProcess:
    """
    Worker process that handles tasks for a specific GPU
    """
    def __init__(self, gpu_id, task_queue, result_queue, model_cache_size=5):
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.process = None
        self.model_cache = {}
        self.model_cache_size = model_cache_size
        self.model_cache_usage = []
        self.shared_memory_blocks = {}

    def start(self):
        """Start the GPU worker process"""
        self.process = multiprocessing.Process(
            target=self._worker_loop,
            args=(self.gpu_id, self.task_queue, self.result_queue, self.model_cache_size),
            daemon=True
        )
        self.process.start()
        return self.process

    def _worker_loop(self, gpu_id, task_queue, result_queue, model_cache_size):
        """Main worker loop that processes tasks from the queue"""
        # Set the GPU for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"GPU Worker {gpu_id} started")
        
        # Import here to ensure we're in the right process
        import comfy.model_management
        import execution
        from server import PromptServer
        
        # Initialize the executor
        server = PromptServer(None)
        executor = execution.PromptExecutor(server)
        
        # Initialize model cache
        model_cache = {}
        model_cache_usage = []
        shared_memory_blocks = {}
        
        try:
            while True:
                try:
                    # Get a task from the queue
                    task = task_queue.get(timeout=1.0)
                    if task is None:  # Poison pill to stop the worker
                        break
                    
                    task_id, prompt_id, prompt, extra_data, outputs_to_execute = task
                    
                    # Check if we need to load models from shared memory
                    self._load_models_from_shared_memory(prompt, model_cache, shared_memory_blocks)
                    
                    # Execute the prompt
                    execution_start_time = time.perf_counter()
                    executor.execute(prompt, prompt_id, extra_data, outputs_to_execute)
                    
                    # Update model cache usage
                    for model_id in model_cache:
                        if model_id in model_cache_usage:
                            model_cache_usage.remove(model_id)
                        model_cache_usage.append(model_id)
                    
                    # Trim cache if needed
                    while len(model_cache_usage) > model_cache_size:
                        model_id = model_cache_usage.pop(0)
                        if model_id in model_cache:
                            del model_cache[model_id]
                    
                    # Send the result back
                    execution_time = time.perf_counter() - execution_start_time
                    result_queue.put((task_id, executor.outputs_ui, {
                        'status_str': 'success' if executor.success else 'error',
                        'completed': executor.success,
                        'execution_time': execution_time,
                        'messages': executor.status_messages
                    }))
                    
                    # Clean up after execution
                    comfy.model_management.cleanup_models()
                    torch.cuda.empty_cache()
                    
                except queue.Empty:
                    # No tasks available, continue waiting
                    continue
                except Exception as e:
                    # Log the error and continue processing other tasks
                    logging.error(f"Error in GPU worker {gpu_id}: {str(e)}")
                    logging.error(traceback.format_exc())
                    try:
                        result_queue.put((task_id, {}, {
                            'status_str': 'error',
                            'completed': False,
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        }))
                    except:
                        pass
        finally:
            # Clean up shared memory blocks
            for shm_name in shared_memory_blocks:
                try:
                    shared_memory_blocks[shm_name].close()
                    shared_memory_blocks[shm_name].unlink()
                except:
                    pass
            logging.info(f"GPU Worker {gpu_id} stopped")

    def _load_models_from_shared_memory(self, prompt, model_cache, shared_memory_blocks):
        """Load models from shared memory if available"""
        # Identify models needed for this prompt
        models_needed = self._identify_models_in_prompt(prompt)
        
        with shared_models_lock:
            for model_id in models_needed:
                if model_id in shared_models and model_id not in model_cache:
                    shm_name = shared_models[model_id]
                    try:
                        # Access the shared memory block
                        shm = shared_memory.SharedMemory(name=shm_name)
                        shared_memory_blocks[shm_name] = shm
                        
                        # Reconstruct the model from shared memory
                        # This is a simplified example - actual implementation would depend on model structure
                        buffer = np.ndarray((shared_models_metadata[model_id]['size'],), 
                                           dtype=np.uint8, buffer=shm.buf)
                        model_data = pickle.loads(buffer.tobytes())
                        
                        # Load the model into the cache
                        model_cache[model_id] = model_data
                        logging.info(f"Loaded model {model_id} from shared memory")
                    except Exception as e:
                        logging.error(f"Error loading model {model_id} from shared memory: {str(e)}")

    def _identify_models_in_prompt(self, prompt):
        """Identify which models are needed for a given prompt"""
        # This is a placeholder - actual implementation would analyze the prompt
        # to determine which models are needed
        models = []
        for node_id, node in prompt.items():
            if 'class_type' in node:
                class_type = node['class_type']
                if 'checkpoint' in class_type.lower() or 'model' in class_type.lower():
                    # Extract model identifier from inputs
                    if 'inputs' in node and 'ckpt_name' in node['inputs']:
                        models.append(node['inputs']['ckpt_name'])
        return models

class MultiGPUManager:
    """
    Manager for distributing tasks across multiple GPUs
    """
    def __init__(self, gpu_ids, model_cache_size=5):
        self.gpu_ids = gpu_ids
        self.model_cache_size = model_cache_size
        self.task_queues = []
        self.result_queue = multiprocessing.Queue()
        self.workers = []
        self.next_task_id = 0
        self.tasks_in_progress = {}
        self.lock = threading.Lock()

    def initialize(self):
        """Initialize the GPU workers"""
        for gpu_id in self.gpu_ids:
            task_queue = multiprocessing.Queue()
            self.task_queues.append(task_queue)
            worker = GPUWorkerProcess(gpu_id, task_queue, self.result_queue, self.model_cache_size)
            self.workers.append(worker)
            worker.start()
        
        # Start the result collector thread
        self.result_collector = threading.Thread(target=self._collect_results, daemon=True)
        self.result_collector.start()
        
        logging.info(f"Multi-GPU manager initialized with GPUs: {self.gpu_ids}")
        return True

    def shutdown(self):
        """Shutdown all GPU workers"""
        for task_queue in self.task_queues:
            task_queue.put(None)  # Send poison pill to stop workers
        
        for worker in self.workers:
            worker.process.join(timeout=5)
            if worker.process.is_alive():
                worker.process.terminate()
        
        logging.info("Multi-GPU manager shutdown complete")

    def submit_task(self, prompt_id, prompt, extra_data=None, outputs_to_execute=None):
        """Submit a task to be executed on one of the GPUs"""
        with self.lock:
            task_id = self.next_task_id
            self.next_task_id += 1
            
            # Find the GPU with the shortest queue
            shortest_queue_idx = 0
            shortest_queue_size = self.task_queues[0].qsize()
            
            for i in range(1, len(self.task_queues)):
                queue_size = self.task_queues[i].qsize()
                if queue_size < shortest_queue_size:
                    shortest_queue_idx = i
                    shortest_queue_size = queue_size
            
            # Create the task
            task = (task_id, prompt_id, prompt, extra_data, outputs_to_execute)
            
            # Add to the tasks in progress
            self.tasks_in_progress[task_id] = {
                'prompt_id': prompt_id,
                'gpu_id': self.gpu_ids[shortest_queue_idx],
                'submit_time': time.time()
            }
            
            # Submit to the queue
            self.task_queues[shortest_queue_idx].put(task)
            
            logging.info(f"Task {task_id} (prompt {prompt_id}) submitted to GPU {self.gpu_ids[shortest_queue_idx]}")
            return task_id

    def _collect_results(self):
        """Collect results from the result queue and process them"""
        while True:
            try:
                result = self.result_queue.get()
                if result is None:  # Poison pill
                    break
                
                task_id, outputs_ui, status = result
                
                with self.lock:
                    if task_id in self.tasks_in_progress:
                        task_info = self.tasks_in_progress.pop(task_id)
                        prompt_id = task_info['prompt_id']
                        gpu_id = task_info['gpu_id']
                        
                        # Process the result
                        # This would typically involve updating the PromptQueue
                        # and notifying clients about the completed task
                        logging.info(f"Task {task_id} (prompt {prompt_id}) completed on GPU {gpu_id}")
                        
                        # Here we would call the appropriate method to handle the completed task
                        # For example: prompt_queue.task_done(prompt_id, outputs_ui, status)
            except Exception as e:
                logging.error(f"Error in result collector: {str(e)}")
                logging.error(traceback.format_exc())

    def share_model(self, model_id, model_data):
        """
        Share a model across processes using shared memory
        
        Args:
            model_id: Unique identifier for the model
            model_data: The model data to share
        """
        try:
            # Serialize the model data
            data = pickle.dumps(model_data)
            data_size = len(data)
            
            # Create shared memory block
            shm = shared_memory.SharedMemory(create=True, size=data_size)
            
            # Copy data to shared memory
            buffer = np.ndarray((data_size,), dtype=np.uint8, buffer=shm.buf)
            buffer[:] = np.frombuffer(data, dtype=np.uint8)
            
            # Register the shared memory block
            with shared_models_lock:
                shared_models[model_id] = shm.name
                shared_models_metadata[model_id] = {
                    'size': data_size,
                    'created': time.time()
                }
            
            logging.info(f"Model {model_id} shared in memory block {shm.name} ({data_size} bytes)")
            return shm.name
        except Exception as e:
            logging.error(f"Error sharing model {model_id}: {str(e)}")
            return None

    def unshare_model(self, model_id):
        """
        Remove a shared model from shared memory
        
        Args:
            model_id: Unique identifier for the model
        """
        with shared_models_lock:
            if model_id in shared_models:
                shm_name = shared_models[model_id]
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                    del shared_models[model_id]
                    del shared_models_metadata[model_id]
                    logging.info(f"Model {model_id} unshared from memory")
                except Exception as e:
                    logging.error(f"Error unsharing model {model_id}: {str(e)}")

# Global shared model metadata
shared_models_metadata = {}

# Create a singleton instance
_instance = None

def get_instance(gpu_ids=None, model_cache_size=5):
    """Get the singleton instance of the MultiGPUManager"""
    global _instance
    if _instance is None and gpu_ids is not None:
        _instance = MultiGPUManager(gpu_ids, model_cache_size)
    return _instance