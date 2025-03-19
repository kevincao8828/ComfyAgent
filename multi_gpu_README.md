# ComfyUI Multi-GPU Support

This extension adds support for running multiple tasks simultaneously across different GPUs in ComfyUI.

## Features

- Run multiple tasks in parallel on different GPUs
- Automatic load balancing to distribute tasks to the least busy GPU
- Complete isolation between GPU workers using separate processes
- Compatible with all existing custom nodes without modification
- Proper cleanup and resource management

## Installation

1. Place the `multi_gpu.py` file in your ComfyUI root directory (same location as `main.py`)
2. Restart ComfyUI with the `--multi-gpu` argument (see Usage below)

## Usage

Launch ComfyUI with the `--multi-gpu` argument followed by a comma-separated list of GPU IDs:

```bash
python main.py --multi-gpu 0,1,2
```

This will enable parallel processing across GPUs 0, 1, and 2.

## How It Works

The multi-GPU implementation uses a process-based approach:

1. Each GPU runs in its own dedicated process with its own CUDA context
2. The main process manages a queue of tasks and distributes them to GPU worker processes
3. Each GPU worker process has its own isolated environment with `CUDA_VISIBLE_DEVICES` set to its specific GPU ID
4. Tasks are automatically assigned to the GPU with the shortest queue
5. Results are sent back to the main process and presented to the user

This approach provides complete isolation between GPUs and ensures compatibility with all existing custom nodes without requiring any modifications to them.

## Advantages Over Thread-Based Approaches

- **Complete Isolation**: Each GPU operates in its own process with its own memory space
- **No GIL Limitations**: Python's Global Interpreter Lock doesn't limit parallel execution
- **Custom Node Compatibility**: All existing custom nodes work without modification
- **Stability**: A crash in one GPU worker doesn't affect others
- **True Parallelism**: Tasks run truly in parallel across different GPUs

## Limitations

- Slightly higher memory usage due to multiple Python processes
- Cannot easily cancel tasks that have already been sent to a GPU worker
- Queue management is more complex

## Troubleshooting

If you encounter issues:

1. Make sure the GPU IDs you specified actually exist on your system
2. Check that you have enough system RAM to run multiple processes
3. Ensure your CUDA installation supports all the GPUs you're trying to use
4. Look for error messages in the console output

## Advanced Configuration

For advanced users, you can modify the `multi_gpu.py` file to customize:

- Memory management policies
- Load balancing algorithms
- Error handling behavior
- Inter-process communication mechanisms

## License

This extension is released under the same license as ComfyUI.