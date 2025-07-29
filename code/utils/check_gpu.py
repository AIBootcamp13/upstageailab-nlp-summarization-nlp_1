

import torch

def check_gpu():
    """
    PyTorch GPU 사용 가능 여부를 확인하고, 사용 가능한 GPU 정보와 MPS 지원 여부를 출력합니다.
    """
    print(f"PyTorch version: {torch.__version__}")

    # CUDA 지원 확인
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {is_cuda_available}")

    if is_cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # MPS (Apple Silicon) 지원 확인
    is_mps_available = torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon) available: {is_mps_available}")
    
    if not is_cuda_available and not is_mps_available:
        print("\nWarning: No GPU acceleration is available. PyTorch will run on CPU.")

if __name__ == "__main__":
    check_gpu()

