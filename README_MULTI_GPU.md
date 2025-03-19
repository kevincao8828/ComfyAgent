# ComfyUI Multi-GPU Support

This extension adds multi-GPU support to ComfyUI, allowing you to run multiple tasks simultaneously across different GPUs. This can significantly increase throughput when processing multiple images or batches.

## Features

- Run multiple tasks in parallel across different GPUs
- Automatic load balancing to distribute tasks to the least busy GPU
- Complete isolation between GPU processes to prevent conflicts
- Compatible with all existing ComfyUI nodes and workflows
- No modifications needed for custom nodes

## Requirements

- Multiple CUDA-compatible GPUs
- Python 3.8 or higher
- ComfyUI and its dependencies

## Installation

No additional installation is required. The multi-GPU support is included in this version of ComfyUI.

## Usage

To use multi-GPU support, launch ComfyUI with the `--multi-gpu` argument followed by a comma-separated list of GPU IDs:

```bash
python main_multi_gpu.py --multi-gpu 0,1,2
```

This will enable parallel processing across GPUs 0, 1, and 2.

### Additional Options

- `--model-cache-size N`: Set the number of models to keep in cache per GPU (default: 3)

## How It Works

The multi-GPU implementation uses separate processes for each GPU, with each process having its own CUDA context and environment. This approach provides several advantages:

1. **Complete Isolation**: Each GPU runs in its own process with its own memory space, preventing conflicts between models and operations.

2. **True Parallelism**: Not limited by Python's Global Interpreter Lock (GIL), allowing true parallel execution.

3. **Compatibility**: No modifications needed for existing custom nodes, as each process sees only one GPU.

4. **Reliability**: If one process crashes, it won't affect the others.

Tasks are automatically distributed to the least busy GPU, ensuring efficient resource utilization.

## Monitoring GPU Status

The system adds a new API endpoint to monitor GPU status:

```
GET /gpu_status
```

This returns a JSON object with the status of each GPU, including whether it's idle or executing a task, and how many tasks are queued for it.

## Limitations

- Each GPU needs enough VRAM to run the models independently
- Shared memory between processes is not implemented in this version
- The web UI doesn't currently show which GPU is processing which task

## Troubleshooting

### Common Issues

1. **"CUDA error: out of memory"**
   - The model is too large for the GPU's VRAM
   - Try using the `--lowvram` option along with `--multi-gpu`

2. **Process crashes on startup**
   - Ensure all GPUs are properly detected by running `nvidia-smi`
   - Check that the GPU IDs specified are valid

3. **One GPU is used more than others**
   - This is normal if tasks have different complexity
   - The load balancer will assign new tasks to less busy GPUs

### Debugging

For more detailed logs, add the `--verbose` flag:

```bash
python main_multi_gpu.py --multi-gpu 0,1,2 --verbose
```

## Performance Tips

1. For best performance, use GPUs of the same model and VRAM capacity
2. If using mixed GPU types, consider assigning specific tasks to specific GPUs
3. Monitor GPU usage with tools like `nvidia-smi` to ensure balanced utilization

## Future Improvements

Planned improvements for future versions:

- Shared memory for model weights to reduce memory usage
- UI enhancements to show GPU assignment and status
- Dynamic VRAM management based on model size and GPU capacity
- Support for specialized task routing based on GPU capabilities