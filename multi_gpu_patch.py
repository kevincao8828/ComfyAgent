import os
import sys
import threading
import time
import logging
import torch
import copy
import asyncio

import comfy.model_management
from comfy.cli_args import args
import execution
import server

# Class to manage a pool of GPU workers
class GPUWorkerPool:
    def __init__(self, server_instance, gpu_ids):
        self.server = server_instance
        self.gpu_ids = gpu_ids
        self.workers = []
        self.queues = []
        self.active = True
        
        logging.info(f"Initializing GPU worker pool with GPUs: {gpu_ids}")
        
        # Create a queue and worker thread for each GPU
        for i, gpu_id in enumerate(gpu_ids):
            # Create a queue for this GPU
            q = execution.PromptQueue(server_instance)
            q.gpu_id = gpu_id  # Add GPU ID to the queue
            self.queues.append(q)
            
            # Create and start a worker thread for this GPU
            worker_thread = threading.Thread(
                target=self.gpu_prompt_worker,
                daemon=True,
                args=(q, server_instance, gpu_id, i)
            )
            self.workers.append(worker_thread)
            worker_thread.start()
            logging.info(f"Started worker thread for GPU {gpu_id}")
    
    def gpu_prompt_worker(self, q, server_instance, gpu_id, worker_id):
        """Worker function that processes tasks for a specific GPU"""
        # Set the CUDA device for this thread
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Create a separate executor for this worker
        e = execution.PromptExecutor(server_instance)
        
        # Variables for garbage collection
        last_gc_collect = 0
        need_gc = False
        gc_collect_interval = 10.0
        
        logging.info(f"Worker {worker_id} using GPU {gpu_id} started")
        
        while self.active:
            try:
                timeout = 1000.0
                if need_gc:
                    current_time = time.perf_counter()
                    timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)
                
                # Get a task from the queue
                queue_item = q.get(timeout=timeout)
                
                if queue_item is not None:
                    item, item_id = queue_item
                    execution_start_time = time.perf_counter()
                    prompt_id = item[1]
                    
                    # Store the prompt ID and set the GPU context
                    server_instance.last_prompt_id = prompt_id
                    
                    # Log which GPU is processing this task
                    logging.info(f"GPU {gpu_id} (Worker {worker_id}) processing prompt {prompt_id}")
                    
                    # Execute the task
                    e.execute(item[2], prompt_id, item[3], item[4])
                    need_gc = True
                    
                    # Mark the task as done
                    q.task_done(item_id,
                                e.outputs_ui,
                                status=execution.PromptQueue.ExecutionStatus(
                                    status_str='success' if e.success else 'error',
                                    completed=e.success,
                                    messages=e.status_messages))
                    
                    if server_instance.client_id is not None:
                        server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id, "gpu_id": gpu_id}, server_instance.client_id)
                    
                    current_time = time.perf_counter()
                    execution_time = current_time - execution_start_time
                    logging.info(f"Prompt executed on GPU {gpu_id} in {:.2f} seconds".format(execution_time))
                
                # Handle flags for memory management
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
                
                # Perform garbage collection if needed
                if need_gc:
                    current_time = time.perf_counter()
                    if (current_time - last_gc_collect) > gc_collect_interval:
                        comfy.model_management.cleanup_models()
                        import gc
                        gc.collect()
                        comfy.model_management.soft_empty_cache()
                        last_gc_collect = current_time
                        need_gc = False
            
            except Exception as ex:
                logging.error(f"Error in worker {worker_id} (GPU {gpu_id}): {ex}")
                logging.error(f"Traceback: {sys.exc_info()}")
    
    def put_task(self, task):
        """Add a task to the least busy queue"""
        # Find the queue with the fewest pending tasks
        min_tasks = float('inf')
        selected_queue = None
        
        for q in self.queues:
            tasks_remaining = q.get_tasks_remaining()
            if tasks_remaining < min_tasks:
                min_tasks = tasks_remaining
                selected_queue = q
        
        # Add the task to the selected queue
        if selected_queue:
            selected_queue.put(task)
            logging.info(f"Task {task[1]} assigned to GPU {selected_queue.gpu_id} (queue size: {min_tasks})")
            return selected_queue.gpu_id
        else:
            logging.error("No queue available to process task")
            return None
    
    def get_all_queue_info(self):
        """Get information about all queues"""
        queue_info = {}
        for i, q in enumerate(self.queues):
            queue_info[f'gpu_{q.gpu_id}'] = {
                'tasks_remaining': q.get_tasks_remaining(),
                'current_tasks': list(q.currently_running.values()),
                'queued_tasks': copy.deepcopy(q.queue)
            }
        return queue_info

# Patch the server's post_prompt method to use the GPU worker pool
def patch_server_post_prompt(server_instance):
    original_post_prompt = server_instance.routes._items[0]._handler
    
    async def patched_post_prompt(request):
        logging.info("got prompt (multi-GPU mode)")
        resp_code = 200
        out_string = ""
        json_data = await request.json()
        if "flowname" not in json_data:
            response = {"prompt_id": prompt_id, "number": number, "node_errors": "no flowname specified"}
            return web.json_response(response)
        flowname = json_data["flowname"]
        json_data = server_instance.trigger_on_prompt(json_data)

        if "number" in json_data:
            number = float(json_data['number'])
        else:
            number = server_instance.number
            if "front" in json_data:
                if json_data['front']:
                    number = -number

            server_instance.number += 1

        if "prompt" in json_data:
            prompt = json_data["prompt"]
            valid = execution.validate_prompt(prompt)
            extra_data = {}
            if "extra_data" in json_data:
                extra_data = json_data["extra_data"]

            if "client_id" in json_data:
                extra_data["client_id"] = json_data["client_id"]
            if valid[0]:
                prompt_id = str(uuid.uuid4())
                outputs_to_execute = valid[2]
                pnginfo = extra_data.get("extra_pnginfo", None)
                if pnginfo is not None:
                    workflow = pnginfo.get("workflow", None)
                    if workflow is not None:
                        save_workflow(flowname, workflow)
                
                # Use the GPU worker pool to distribute the task
                task = (number, prompt_id, prompt, extra_data, outputs_to_execute)
                gpu_id = server_instance.gpu_worker_pool.put_task(task)
                
                response = {
                    "prompt_id": prompt_id, 
                    "number": number, 
                    "node_errors": valid[3],
                    "gpu_id": gpu_id
                }
                return web.json_response(response)
            else:
                logging.warning("invalid prompt: {}".format(valid[1]))
                return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
        else:
            return web.json_response({"error": "no prompt", "node_errors": []}, status=400)
    
    # Replace the original handler with our patched version
    for route in server_instance.routes._items:
        if route.method == "POST" and route.path == "/prompt":
            route._handler = patched_post_prompt
            break

# Patch the server's get_queue_info method to include all GPU queues
def patch_server_get_queue_info(server_instance):
    original_get_queue_info = server_instance.get_queue_info
    
    def patched_get_queue_info():
        prompt_info = {}
        exec_info = {}
        
        # Get information from all GPU queues
        gpu_queues = server_instance.gpu_worker_pool.get_all_queue_info()
        exec_info['gpu_queues'] = gpu_queues
        
        # Calculate total remaining tasks across all queues
        total_remaining = sum(info['tasks_remaining'] for info in gpu_queues.values())
        exec_info['queue_remaining'] = total_remaining
        
        prompt_info['exec_info'] = exec_info
        return prompt_info
    
    # Replace the original method with our patched version
    server_instance.get_queue_info = patched_get_queue_info

# Main function to apply the multi-GPU patch
def apply_multi_gpu_patch(server_instance, gpu_ids):
    # Create the GPU worker pool
    server_instance.gpu_worker_pool = GPUWorkerPool(server_instance, gpu_ids)
    
    # Patch the server methods
    patch_server_post_prompt(server_instance)
    patch_server_get_queue_info(server_instance)
    
    # Add a new route to get GPU status
    @server_instance.routes.get("/gpu_status")
    async def get_gpu_status(request):
        status = {}
        for i, gpu_id in enumerate(server_instance.gpu_worker_pool.gpu_ids):
            q = server_instance.gpu_worker_pool.queues[i]
            status[f'gpu_{gpu_id}'] = {
                'tasks_remaining': q.get_tasks_remaining(),
                'current_tasks': [t[1] for t in q.currently_running.values()],  # Just get prompt IDs
            }
        return web.json_response(status)
    
    logging.info("Multi-GPU patch applied successfully")