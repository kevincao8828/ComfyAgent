# ComfyUI Multi-GPU Support

This implementation adds support for running multiple tasks simultaneously across different GPUs in ComfyUI, with shared memory optimization to reduce memory usage.

## Features

- Run multiple tasks in parallel across different GPUs
- Automatic load balancing to distribute tasks to the least busy GPU
- Shared memory for model weights to reduce memory usage
- Model caching to avoid reloading models frequently
- Minimal changes to the core ComfyUI codebase
- Compatible with all existing custom nodes

## Installation

1. Copy the `multi_gpu.py` file to your ComfyUI directory
2. Use `main_multi_gpu.py` instead of `main.py` to start ComfyUI

## Usage

Launch ComfyUI with the `--multi-gpu` argument followed by a comma-separated list of GPU IDs:

```bash
python main_multi_gpu.py --multi-gpu 0,1,2
```

This will enable parallel processing across GPUs 0, 1, and 2.

You can also specify the number of models to keep in cache per GPU:

```bash
python main_multi_gpu.py --multi-gpu 0,1,2 --model-cache-size 3
```

## How It Works

### Architecture

The multi-GPU implementation uses a process-based approach where:

1. Each GPU runs in its own dedicated process with its own CUDA context
2. A main process manages task distribution and result collection
3. Tasks are automatically assigned to the GPU with the shortest queue
4. Results are collected and sent back to the main process

### Shared Memory Optimization

To reduce memory usage, the implementation uses Python's `multiprocessing.shared_memory` to share model weights between processes:

1. When a model is loaded for the first time, it's stored in shared memory
2. Subsequent processes can access the model from shared memory instead of loading it again
3. This significantly reduces the total memory usage when the same model is used across multiple GPUs

### Model Caching

Each GPU worker maintains a cache of recently used models:

1. Models are kept in memory based on a Least Recently Used (LRU) policy
2. The cache size can be configured with the `--model-cache-size` parameter
3. This prevents models from being repeatedly loaded and unloaded

## Limitations

1. The shared memory implementation works best for models that can be easily serialized
2. Some custom nodes might not be fully compatible with the shared memory approach
3. The implementation requires Python 3.8+ for shared memory support

## Troubleshooting

If you encounter issues with the multi-GPU implementation:

1. Check that all specified GPUs are available and have enough memory
2. Ensure that your CUDA drivers are up to date
3. Try reducing the model cache size if you're running out of memory
4. If a specific custom node causes issues, try running it in single-GPU mode

## Fallback Mechanism

If the multi-GPU initialization fails for any reason, the system will automatically fall back to single-GPU mode, ensuring that your workflows can still run.

## Performance Tips

1. For best performance, distribute tasks across GPUs with similar specifications
2. If you have GPUs with different memory capacities, assign more memory-intensive tasks to the GPUs with more memory
3. Monitor GPU usage to ensure balanced utilization

## Future Improvements

1. Dynamic load balancing based on GPU memory and utilization
2. More sophisticated model sharing mechanisms
3. Support for distributed execution across multiple machines