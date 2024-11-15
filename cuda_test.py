import cv2
import numpy as np
import torch
import sys

def print_cuda_info():
    print("\n=== CUDA System Information ===")

    print("\n--- OpenCV Information ---")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    print(f"Available CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("\n--- CUDA Device Properties ---")
        dev = cv2.cuda.getDevice()
        print(f"Current device ID: {dev}")

        # Get all available CUDA functions and classes
        print("\n--- Available CUDA Functions in OpenCV ---")
        cuda_functions = [attr for attr in dir(cv2.cuda) if not attr.startswith('_')]
        for func in cuda_functions:
            print(f"cv2.cuda.{func}")

    print("\n--- PyTorch CUDA Information ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability()}")
        print(f"Device properties: {torch.cuda.get_device_properties(0)}")

    try:
        import cupy as cp
        print("\n--- CuPy Information ---")
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"Available devices: {cp.cuda.runtime.getDeviceCount()}")
    except ImportError:
        print("\nCuPy not installed")

    # Test basic CUDA operations
    print("\n=== Testing Basic CUDA Operations ===")

    print("\n--- Testing OpenCV CUDA ---")
    try:
        # Try to create a CUDA GpuMat
        test_array = np.random.rand(100, 100).astype(np.float32)
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(test_array)
        result = gpu_mat.download()
        print("OpenCV CUDA GpuMat test: SUCCESS")

        # List available OpenCV CUDA algorithms
        print("\nAvailable OpenCV CUDA algorithms:")
        for name in dir(cv2.cuda):
            if not name.startswith('_'):
                obj = getattr(cv2.cuda, name)
                if isinstance(obj, type):
                    print(f"- {name}")
    except Exception as e:
        print(f"OpenCV CUDA test failed: {e}")

    print("\n--- Testing PyTorch CUDA ---")
    if torch.cuda.is_available():
        try:
            x = torch.rand(100, 100).cuda()
            y = x + x
            print("PyTorch CUDA tensor test: SUCCESS")
        except Exception as e:
            print(f"PyTorch CUDA test failed: {e}")

    print("\n=== Memory Information ===")
    if torch.cuda.is_available():
        print("\nPyTorch CUDA Memory:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    print_cuda_info()
