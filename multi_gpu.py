import os
import sys
import time
import json
import logging
import threading
import multiprocessing
from multiprocessing import Process, Queue, Event
import torch

import execution
import server
from nodes import interrupt_processing
import comfy.model_management

class GPUWorkerProcess:
    """
    Worker process that runs on a specific GPU
    """
    def __init__(self, gpu_id, task_queue, result_queue, stop_event):
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.process = None
    
    def start(self):
        self.process = Process(
            target=self._worker_loop,
            args=(self.gpu_id, self.task_queue, self.result_queue, self.stop_event)
        )
        self.process.daemon = True
        self.process.start()
        logging.info(f"Started GPU worker process for GPU {self.gpu_id} (PID: {self.process.pid})")
    
    def terminate(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            logging.info(f"Terminated GPU worker process for GPU {self.gpu_id}")
    
    @staticmethod
    def _worker_loop(gpu_id, task_queue, result_queue, stop_event):
        # Set the CUDA_VISIBLE_DEVICES for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"GPU worker process for GPU {gpu_id} started with CUDA_VISIBLE_DEVICES={gpu_id}")
        
        # Initialize server and executor for this process
        loop = None
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create a server instance for this worker
            worker_server = server.PromptServer(loop)
            worker_server.add_routes()
            
            # Create an executor for this worker
            executor = execution.PromptExecutor(worker_server)
            
            # Process tasks until stop event is set
            while not stop_event.is_set():
                try:
                    # Try to get a task with a timeout to allow checking the stop event
                    try:
                        task = task_queue.get(timeout=1.0)
                    except:
                        continue
                    
                    if task is None:
                        # None is a signal to exit
                        break
                    
                    # Extract task data
                    task_id, prompt_id, prompt, extra_data, outputs_to_execute = task
                    
                    # Execute the prompt
                    execution_start_time = time.perf_counter()
                    executor.execute(prompt, prompt_id, extra_data, outputs_to_execute)
                    execution_time = time.perf_counter() - execution_start_time
                    
                    # Send the result back
                    result = {
                        'task_id': task_id,
                        'prompt_id': prompt_id,
                        'success': executor.success,
                        'outputs_ui': executor.outputs_ui,
                        'status': {
                            'status_str': 'success' if executor.success else 'error',
                            'completed': executor.success,
                            'messages': executor.status_messages
                        },
                        'execution_time': execution_time,
                        'gpu_id': gpu_id
                    }
                    result_queue.put(result)
                    
                    # Clean up after task execution
                    comfy.model_management.unload_all_models()
                    comfy.model_management.cleanup_models()
                    torch.cuda.empty_cache()
                    
                    logging.info(f"GPU {gpu_id}: Prompt executed in {execution_time:.2f} seconds")
                    
                except Exception as e:
                    logging.error(f"GPU {gpu_id}: Error in worker process: {str(e)}")
                    logging.error(f"GPU {gpu_id}: {sys.exc_info()[0].__name__}: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
                    
                    # Try to send error result
                    try:
                        error_result = {
                            'task_id': task_id if 'task_id' in locals() else -1,
                            'prompt_id': prompt_id if 'prompt_id' in locals() else "unknown",
                            'success': False,
                            'outputs_ui': {},
                            'status': {
                                'status_str': 'error',
                                'completed': False,
                                'messages': [(
                                    "execution_error", 
                                    {
                                        "exception_message": str(e),
                                        "exception_type": sys.exc_info()[0].__name__,
                                        "traceback": traceback.format_exc()
                                    },
                                    False
                                )]
                            },
                            'execution_time': 0,
                            'gpu_id': gpu_id
                        }
                        result_queue.put(error_result)
                    except:
                        pass
        
        except Exception as e:
            logging.error(f"GPU {gpu_id}: Fatal error in worker process: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        
        finally:
            # Clean up
            logging.info(f"GPU {gpu_id}: Worker process exiting")
            if loop:
                loop.close()


class MultiGPUManager:
    """
    Manages multiple GPU worker processes and distributes tasks among them
    """
    def __init__(self, gpu_ids, server_instance):
        self.gpu_ids = gpu_ids
        self.server = server_instance
        self.workers = []
        self.task_queues = []
        self.result_queue = Queue()
        self.stop_event = Event()
        self.task_counter = 0
        self.tasks_in_progress = {}
        self.result_thread = None
        
        # Create a task queue for each GPU
        for _ in self.gpu_ids:
            self.task_queues.append(Queue())
        
        # Create and start worker processes
        for i, gpu_id in enumerate(self.gpu_ids):
            worker = GPUWorkerProcess(
                gpu_id=gpu_id,
                task_queue=self.task_queues[i],
                result_queue=self.result_queue,
                stop_event=self.stop_event
            )
            self.workers.append(worker)
            worker.start()
        
        # Start result handling thread
        self.result_thread = threading.Thread(target=self._handle_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        logging.info(f"Multi-GPU manager initialized with GPUs: {self.gpu_ids}")
    
    def submit_task(self, prompt_tuple):
        """
        Submit a task to be executed on one of the GPUs
        prompt_tuple: (priority_number, prompt_id, prompt, extra_data, outputs_to_execute)
        """
        # Assign a task ID
        task_id = self.task_counter
        self.task_counter += 1
        
        # Find the GPU with the shortest queue
        shortest_queue_idx = 0
        shortest_queue_size = self.task_queues[0].qsize()
        
        for i in range(1, len(self.task_queues)):
            queue_size = self.task_queues[i].qsize()
            if queue_size < shortest_queue_size:
                shortest_queue_idx = i
                shortest_queue_size = queue_size
        
        # Store task in progress
        self.tasks_in_progress[task_id] = {
            'prompt_id': prompt_tuple[1],
            'gpu_idx': shortest_queue_idx,
            'gpu_id': self.gpu_ids[shortest_queue_idx],
            'submitted_time': time.time()
        }
        
        # Submit the task to the selected GPU
        self.task_queues[shortest_queue_idx].put((task_id,) + prompt_tuple[1:])
        
        logging.info(f"Submitted task {prompt_tuple[1]} to GPU {self.gpu_ids[shortest_queue_idx]} (queue size: {shortest_queue_size})")
        
        return task_id
    
    def _handle_results(self):
        """
        Thread that handles results from worker processes
        """
        while not self.stop_event.is_set():
            try:
                # Get result with timeout to allow checking stop_event
                try:
                    result = self.result_queue.get(timeout=1.0)
                except:
                    continue
                
                task_id = result['task_id']
                prompt_id = result['prompt_id']
                
                if task_id in self.tasks_in_progress:
                    # Remove task from in-progress list
                    task_info = self.tasks_in_progress.pop(task_id)
                    
                    # Notify the server that the task is done
                    if self.server and self.server.prompt_queue:
                        status = execution.PromptQueue.ExecutionStatus(
                            status_str=result['status']['status_str'],
                            completed=result['status']['completed'],
                            messages=result['status']['messages']
                        )
                        
                        self.server.prompt_queue.task_done(
                            task_id,
                            result['outputs_ui'],
                            status=status
                        )
                        
                        # If client_id is in extra_data, send completion notification
                        if 'client_id' in task_info.get('extra_data', {}):
                            client_id = task_info['extra_data']['client_id']
                            self.server.send_sync(
                                "executing", 
                                {"node": None, "prompt_id": prompt_id}, 
                                client_id
                            )
                    
                    gpu_id = result.get('gpu_id', 'unknown')
                    execution_time = result.get('execution_time', 0)
                    logging.info(f"Task {prompt_id} completed on GPU {gpu_id} in {execution_time:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Error in result handler: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
    
    def shutdown(self):
        """
        Shutdown all worker processes
        """
        logging.info("Shutting down Multi-GPU manager")
        self.stop_event.set()
        
        # Send termination signal to all workers
        for queue in self.task_queues:
            try:
                queue.put(None)  # None is the signal to exit
            except:
                pass
        
        # Wait for result thread to finish
        if self.result_thread and self.result_thread.is_alive():
            self.result_thread.join(timeout=5)
        
        # Terminate all worker processes
        for worker in self.workers:
            worker.terminate()
        
        logging.info("Multi-GPU manager shutdown complete")


class MultiGPUPromptQueue(execution.PromptQueue):
    """
    A version of PromptQueue that distributes tasks to multiple GPUs
    """
    def __init__(self, server, gpu_ids):
        super().__init__(server)
        self.gpu_manager = MultiGPUManager(gpu_ids, server)
        logging.info(f"Initialized Multi-GPU prompt queue with GPUs: {gpu_ids}")
    
    def put(self, item):
        with self.mutex:
            # Submit the task to the GPU manager
            task_id = self.gpu_manager.submit_task(item)
            
            # Store the task in currently_running
            self.currently_running[task_id] = item
            
            # Update the server
            self.server.queue_updated()
    
    def get(self, timeout=None):
        # This method is not used in multi-GPU mode
        # Tasks are directly submitted to GPU workers in put()
        return None
    
    def wipe_queue(self):
        # Cannot easily wipe queues in multi-GPU mode
        # Just update the server to reflect that the queue is empty
        with self.mutex:
            self.server.queue_updated()
    
    def delete_queue_item(self, function):
        # Not implemented for multi-GPU mode
        return False
    
    def shutdown(self):
        """
        Shutdown the GPU manager
        """
        if hasattr(self, 'gpu_manager'):
            self.gpu_manager.shutdown()


def parse_gpu_ids(gpu_arg):
    """
    Parse the --multi-gpu argument into a list of GPU IDs
    """
    try:
        if ',' in gpu_arg:
            # Comma-separated list of GPU IDs
            return [int(gpu_id.strip()) for gpu_id in gpu_arg.split(',')]
        else:
            # Single GPU ID
            return [int(gpu_arg.strip())]
    except ValueError:
        logging.error(f"Invalid GPU ID format: {gpu_arg}. Expected comma-separated integers.")
        return [0]  # Default to GPU 0