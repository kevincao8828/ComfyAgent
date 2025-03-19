import comfy.options
comfy.options.enable_args_parsing()

import os
import sys
import importlib.util
import folder_paths
import time
import argparse
import multiprocessing
import threading
import queue
import json
import copy
import logging
import asyncio
import itertools
import shutil
import gc
import torch
import uuid

from comfy.cli_args import args, parser
from comfy.cli_args import EnumAction, LatentPreviewMethod
import comfy.utils
import comfy.model_management

import execution
import server
from server import BinaryEventTypes
from nodes import init_custom_nodes
import yaml

# Add multi-GPU argument
parser.add_argument("--multi-gpu", type=str, help="Comma-separated list of GPU IDs to use for parallel processing (e.g., '0,1,2')")
parser.add_argument("--model-cache-size", type=int, default=3, help="Number of models to keep in cache per GPU")

# Parse arguments again with the new option
if comfy.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

# Setup logging
logging_level = logging.INFO
if args.verbose:
    logging_level = logging.DEBUG

logging.basicConfig(format="%(message)s", level=logging_level)

# Execute pre-startup scripts
def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            print(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        print("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()

execute_prestartup_script()

def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")

def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                logging.info("Adding extra search path {} {}".format(x, full_path))
                folder_paths.add_model_folder_path(x, full_path)

def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

# GPU Worker Process
def gpu_worker_process(gpu_id, input_queue, output_queue, status_queue):
    try:
        # Set environment variable for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"GPU Worker {gpu_id} started")
        
        # Initialize executor
        executor = execution.PromptExecutor(None)
        
        while True:
            try:
                # Get task from queue
                task = input_queue.get()
                
                # Check for termination signal
                if task is None:
                    break
                
                # Extract task data
                task_id, prompt_id, prompt, extra_data, outputs_to_execute = task
                
                # Update status
                status_queue.put(("executing", gpu_id, prompt_id))
                
                # Execute the prompt
                execution_start_time = time.perf_counter()
                
                # Set the server client_id if provided
                if "client_id" in extra_data:
                    executor.server = type('obj', (object,), {
                        'client_id': extra_data["client_id"],
                        'last_node_id': None,
                        'last_prompt_id': prompt_id,
                        'send_sync': lambda *args: None  # Dummy function
                    })
                
                # Execute with inference mode
                with torch.inference_mode():
                    executor.execute(prompt, prompt_id, extra_data, outputs_to_execute)
                
                # Calculate execution time
                execution_time = time.perf_counter() - execution_start_time
                logging.info(f"GPU {gpu_id}: Prompt executed in {:.2f} seconds".format(execution_time))
                
                # Send results back
                output_queue.put((task_id, prompt_id, executor.outputs_ui, executor.success, executor.status_messages))
                
                # Clean up
                comfy.model_management.cleanup_models()
                gc.collect()
                comfy.model_management.soft_empty_cache()
                
                # Update status
                status_queue.put(("idle", gpu_id, None))
                
            except Exception as e:
                logging.error(f"GPU {gpu_id} worker error: {str(e)}")
                logging.error(f"Traceback: {sys.exc_info()}")
                # Try to notify about the error
                if 'prompt_id' in locals():
                    output_queue.put((task_id, prompt_id, {}, False, [(
                        "execution_error", 
                        {"prompt_id": prompt_id, "exception_message": str(e)}
                    )]))
                status_queue.put(("idle", gpu_id, None))
    
    except Exception as e:
        logging.error(f"GPU {gpu_id} process crashed: {str(e)}")
        logging.error(f"Traceback: {sys.exc_info()}")
    
    finally:
        logging.info(f"GPU Worker {gpu_id} shutting down")

# Multi-GPU Queue Manager
class MultiGPUPromptQueue(execution.PromptQueue):
    def __init__(self, server, gpu_ids):
        super().__init__(server)
        self.gpu_ids = gpu_ids
        self.gpu_count = len(gpu_ids)
        self.gpu_status = {gpu_id: "idle" for gpu_id in gpu_ids}
        self.gpu_tasks = {gpu_id: 0 for gpu_id in gpu_ids}
        
        # Create queues for each GPU
        self.input_queues = {gpu_id: multiprocessing.Queue() for gpu_id in gpu_ids}
        self.output_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()
        
        # Start GPU worker processes
        self.processes = {}
        for gpu_id in gpu_ids:
            p = multiprocessing.Process(
                target=gpu_worker_process,
                args=(gpu_id, self.input_queues[gpu_id], self.output_queue, self.status_queue),
                daemon=True
            )
            p.start()
            self.processes[gpu_id] = p
        
        # Start result collector thread
        self.result_thread = threading.Thread(target=self._collect_results, daemon=True)
        self.result_thread.start()
        
        # Start status collector thread
        self.status_thread = threading.Thread(target=self._collect_status, daemon=True)
        self.status_thread.start()
        
        logging.info(f"Multi-GPU system initialized with GPUs: {gpu_ids}")
    
    def _collect_results(self):
        """Thread to collect results from GPU workers"""
        while True:
            try:
                # Get result from any GPU
                task_id, prompt_id, outputs_ui, success, status_messages = self.output_queue.get()
                
                # Find which GPU was running this task
                for gpu_id in self.gpu_ids:
                    if self.gpu_tasks.get(task_id) == gpu_id:
                        self.gpu_tasks.pop(task_id, None)
                        break
                
                # Process the result
                with self.mutex:
                    if task_id in self.currently_running:
                        prompt = self.currently_running.pop(task_id)
                        
                        # Add to history
                        if len(self.history) > execution.MAXIMUM_HISTORY_SIZE:
                            self.history.pop(next(iter(self.history)))
                        
                        status_dict = None
                        if success is not None:
                            status_dict = {
                                "status_str": "success" if success else "error",
                                "completed": success,
                                "messages": status_messages
                            }
                        
                        self.history[prompt_id] = {
                            "prompt": prompt,
                            "outputs": copy.deepcopy(outputs_ui),
                            "status": status_dict,
                        }
                        
                        # Notify server
                        self.server.queue_updated()
                        
                        # Send status messages to client
                        if self.server.client_id is not None:
                            for event, data in status_messages:
                                self.server.send_sync(event, data, self.server.client_id)
            
            except Exception as e:
                logging.error(f"Result collector error: {str(e)}")
                logging.error(f"Traceback: {sys.exc_info()}")
    
    def _collect_status(self):
        """Thread to collect status updates from GPU workers"""
        while True:
            try:
                status, gpu_id, prompt_id = self.status_queue.get()
                self.gpu_status[gpu_id] = status
                
                # Log status change
                if status == "executing" and prompt_id is not None:
                    logging.info(f"GPU {gpu_id} is now executing prompt {prompt_id}")
                elif status == "idle":
                    logging.info(f"GPU {gpu_id} is now idle")
                
                # Notify server of status change
                self.server.queue_updated()
            
            except Exception as e:
                logging.error(f"Status collector error: {str(e)}")
    
    def put(self, item):
        """Add a task to the queue"""
        with self.mutex:
            # Add to our queue
            heapq.heappush(self.queue, item)
            self.server.queue_updated()
            self.not_empty.notify()
    
    def get(self, timeout=None):
        """Get a task from the queue and assign to least busy GPU"""
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait(timeout=timeout)
                if timeout is not None and len(self.queue) == 0:
                    return None
            
            # Get next item from queue
            item = heapq.heappop(self.queue)
            
            # Find least busy GPU
            least_busy_gpu = min(self.gpu_ids, key=lambda gpu_id: 
                                 (0 if self.gpu_status[gpu_id] == "idle" else 1, 
                                  len([1 for g in self.gpu_tasks.values() if g == gpu_id])))
            
            # Assign task ID
            i = self.task_counter
            self.currently_running[i] = copy.deepcopy(item)
            self.gpu_tasks[i] = least_busy_gpu
            self.task_counter += 1
            
            # Send to GPU worker
            self.input_queues[least_busy_gpu].put((i, item[1], item[2], item[3], item[4]))
            
            # Update server
            self.server.queue_updated()
            
            return (item, i)
    
    def task_done(self, item_id, outputs, status=None):
        """This is handled by the result collector thread"""
        pass
    
    def get_gpu_status(self):
        """Get status of all GPUs"""
        return {gpu_id: {
            "status": self.gpu_status[gpu_id],
            "tasks": len([1 for g in self.gpu_tasks.values() if g == gpu_id])
        } for gpu_id in self.gpu_ids}
    
    def shutdown(self):
        """Shutdown all GPU workers"""
        for gpu_id in self.gpu_ids:
            try:
                self.input_queues[gpu_id].put(None)  # Send termination signal
                self.processes[gpu_id].join(timeout=5)
                if self.processes[gpu_id].is_alive():
                    self.processes[gpu_id].terminate()
            except Exception as e:
                logging.error(f"Error shutting down GPU {gpu_id}: {str(e)}")

# Extend server to include GPU status
class MultiGPUPromptServer(server.PromptServer):
    def __init__(self, loop):
        super().__init__(loop)
        
        # Add route for GPU status
        @self.routes.get("/gpu_status")
        async def get_gpu_status(request):
            if hasattr(self.prompt_queue, 'get_gpu_status'):
                return web.json_response(self.prompt_queue.get_gpu_status())
            else:
                return web.json_response({})

# Main function
async def run_multi_gpu():
    # Parse GPU IDs
    if args.multi_gpu:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.multi_gpu.split(',')]
    else:
        logging.error("No GPU IDs specified. Use --multi-gpu argument with comma-separated GPU IDs.")
        return
    
    # Set up event loop
    loop = asyncio.get_event_loop()
    
    # Create server
    server_instance = MultiGPUPromptServer(loop)
    
    # Create multi-GPU queue
    q = MultiGPUPromptQueue(server_instance, gpu_ids)
    
    # Load extra model paths
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)
    
    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)
    
    # Initialize custom nodes
    init_custom_nodes()
    
    # Add routes and set up server
    server_instance.add_routes()
    
    # Set up progress hook
    def hijack_progress(server):
        def hook(value, total, preview_image):
            comfy.model_management.throw_exception_if_processing_interrupted()
            progress = {"value": value, "max": total, "prompt_id": server.last_prompt_id, "node": server.last_node_id}
            
            server.send_sync("progress", progress, server.client_id)
            if preview_image is not None:
                server.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server.client_id)
        comfy.utils.set_progress_bar_global_hook(hook)
    
    hijack_progress(server_instance)
    
    # Set up directories
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()
    
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)
    
    # Set up default folders
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    
    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)
    
    # Auto-launch setup
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server
    
    # Start server
    try:
        await asyncio.gather(
            server_instance.start(address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start),
            server_instance.publish_loop()
        )
    except KeyboardInterrupt:
        logging.info("\nStopped server")
    finally:
        # Shutdown GPU workers
        if hasattr(q, 'shutdown'):
            q.shutdown()
        
        # Clean up
        cleanup_temp()

if __name__ == "__main__":
    if args.cuda_device is not None:
        logging.warning("Warning: --cuda-device is ignored when using --multi-gpu")
    
    if args.multi_gpu:
        # Import cuda_malloc here to avoid circular imports
        import cuda_malloc
        
        # Run the multi-GPU server
        asyncio.run(run_multi_gpu())
    else:
        logging.error("No GPU IDs specified. Use --multi-gpu argument with comma-separated GPU IDs.")
        logging.info("Example: python main_multi_gpu.py --multi-gpu 0,1,2")
        sys.exit(1)