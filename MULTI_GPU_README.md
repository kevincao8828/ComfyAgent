# Multi-GPU Support for ComfyUI

This patch adds support for using multiple GPUs simultaneously in ComfyUI, allowing you to run multiple tasks in parallel across different GPUs.

## Features

- Run multiple tasks simultaneously on different GPUs
- Automatic load balancing across available GPUs
- GPU-specific task queues with independent memory management
- Status monitoring for each GPU

## Installation

1. Place the `multi_gpu_patch.py` file in your ComfyUI root directory (the same directory as `main.py`).
2. The patch is automatically applied when you use the `--multi-gpu` command-line argument.

## Usage

Launch ComfyUI with the `--multi-gpu` argument followed by a comma-separated list of GPU IDs:

```bash
python main.py --multi-gpu 0,1,2
```

This will enable multi-GPU mode using GPUs 0, 1, and 2.

## How It Works

1. **Task Distribution**: When you submit a workflow, the system automatically assigns it to the GPU with the shortest queue.
2. **Isolated Execution**: Each GPU has its own worker thread and execution environment, ensuring that tasks don't interfere with each other.
3. **Memory Management**: Each GPU worker manages its own memory independently, with proper cleanup between tasks.

## API Endpoints

The patch adds a new API endpoint for monitoring GPU status:

- `/gpu_status`: Returns the current status of all GPUs, including tasks in progress and queue lengths.

## Limitations

- Each task runs on a single GPU (no task splitting across multiple GPUs)
- Models are loaded independently on each GPU, which may increase memory usage
- Some custom nodes might not be fully compatible with multi-GPU mode

## Troubleshooting

If you encounter issues with multi-GPU mode:

1. Check that all specified GPUs are available and have enough memory
2. Try running with the `--verbose` flag for more detailed logs
3. If a specific workflow fails, try running it in single-GPU mode to isolate the issue

If multi-GPU mode fails to initialize, the system will automatically fall back to single-GPU mode.

## Advanced Configuration

For systems with GPUs of different capabilities, you might want to adjust the scheduling algorithm in `multi_gpu_patch.py`. The current implementation uses a simple "shortest queue" approach, but you could modify it to consider GPU memory, compute capability, or other factors.