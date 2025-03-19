# ComfyUI Multi-GPU Support

This feature allows ComfyUI to utilize multiple GPUs simultaneously, significantly increasing throughput for batch processing tasks.

## Features

- Run multiple tasks in parallel across different GPUs
- Automatic load balancing to distribute tasks to the least busy GPU
- Proper memory isolation between GPU workers
- Seamless integration with the existing ComfyUI interface
- Fallback to single-GPU mode if needed

## Requirements

- Multiple CUDA-compatible GPUs
- Python 3.8 or higher
- PyTorch with CUDA support

## Usage

To enable multi-GPU support, launch ComfyUI with the `--multi-gpu` argument followed by a comma-separated list of GPU IDs:

```bash
python main.py --multi-gpu 0,1,2
```

This will enable parallel processing across GPUs 0, 1, and 2.

## How It Works

The multi-GPU implementation uses a process-based approach:

1. Each GPU runs in its own dedicated process with its own CUDA context
2. The main process distributes tasks to worker processes based on availability
3. Each worker process sets its own `CUDA_VISIBLE_DEVICES` environment variable
4. Results are communicated back to the main process and presented in the UI

This approach ensures proper isolation between GPU contexts and avoids the issues that would occur with a thread-based approach.

## Limitations

- Each GPU needs enough VRAM to run the models independently
- Some operations that require synchronization between GPUs may not be optimized
- Custom nodes that make assumptions about GPU availability might need modifications

## Troubleshooting

If you encounter issues with multi-GPU mode:

1. Make sure all specified GPUs are available and have CUDA support
2. Check that you have enough system RAM to support multiple GPU workers
3. Try with fewer GPUs if you experience stability issues
4. Fall back to single-GPU mode if necessary

## Performance Tips

- For best performance, use GPUs with similar specifications
- Monitor GPU usage to ensure balanced workloads
- Consider using `--lowvram` for each GPU if running large models
- Adjust batch sizes based on the number of available GPUs

## Future Improvements

- Dynamic worker scaling based on system load
- More sophisticated load balancing algorithms
- GPU-specific configuration options
- Performance metrics and monitoring