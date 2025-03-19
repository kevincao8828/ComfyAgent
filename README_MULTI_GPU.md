# ComfyUI Multi-GPU Support

This implementation adds support for running multiple tasks simultaneously across different GPUs in ComfyUI.

## Features

- Run multiple tasks in parallel across different GPUs
- Automatic load balancing to distribute tasks to the least busy GPU
- Complete isolation between GPU contexts using `torch.cuda.set_device()`
- Compatible with all existing custom nodes without any modifications

## Why use `torch.cuda.set_device()` instead of `CUDA_VISIBLE_DEVICES`?

This implementation uses `torch.cuda.set_device()` to select which GPU each worker thread uses, rather than setting the `CUDA_VISIBLE_DEVICES` environment variable. This approach has several advantages:

1. **Thread-level GPU selection**: Each worker thread can select its own GPU without affecting other threads
2. **No environment variable conflicts**: Environment variables are process-level, not thread-level, so modifying them in threads can cause conflicts
3. **Compatible with custom nodes**: Works with existing custom nodes that use `torch.cuda.current_device()`
4. **No need to restart processes**: Can dynamically switch between GPUs without restarting

## How to Use

Simply launch ComfyUI with the new script and specify which GPUs to use:

```bash
python main_multi_gpu.py --multi-gpu 0,1,2
```

This will enable parallel processing across GPUs 0, 1, and 2, significantly increasing throughput for batch processing.

## Implementation Details

1. Each GPU has its own worker thread that uses `torch.cuda.set_device()` to select the appropriate GPU
2. Tasks are automatically distributed to the GPU with the shortest queue
3. Each worker thread has its own `PromptExecutor` instance for executing tasks
4. The main queue keeps track of all tasks for history and UI purposes

## Limitations

1. Memory is not shared between GPUs, so each GPU needs to load its own copy of models
2. Some operations that explicitly specify a device (e.g., `torch.device("cuda:0")`) may override the current device setting
3. Very large models might still require a single GPU with sufficient VRAM

## Troubleshooting

If you encounter issues:

1. Make sure all specified GPUs are available and have enough memory
2. Check that your custom nodes don't explicitly specify GPU devices
3. Try running with a single GPU first to ensure your workflow works correctly
4. Monitor GPU memory usage to ensure tasks are being properly distributed

## Future Improvements

Potential future enhancements:

1. Implement model sharing between GPUs using shared memory
2. Add GPU-specific memory management settings
3. Implement task prioritization based on GPU capabilities
4. Add dynamic GPU allocation based on task requirements