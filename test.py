#!/usr/bin/env python3
"""
GPU Status Checker
Checks the status of all available GPUs (up to 8)
"""

import subprocess
import sys

def check_gpus_with_torch():
    """Check GPU status using PyTorch"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA is not available")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s)\n")
        
        for i in range(min(gpu_count, 8)):
            try:
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = props.total_memory / 1024**3
                
                print(f"GPU {i}: {props.name}")
                print(f"  └─ Total Memory: {memory_total:.2f} GB")
                print(f"  └─ Allocated: {memory_allocated:.2f} GB")
                print(f"  └─ Reserved: {memory_reserved:.2f} GB")
                print(f"  └─ Available: {memory_total - memory_reserved:.2f} GB")
                print(f"  └─ Compute Capability: {props.major}.{props.minor}")
                print()
            except Exception as e:
                print(f"GPU {i}: ⚠️  Error reading properties - {e}\n")
        
        return True
    except ImportError:
        return False

def check_gpus_with_nvidia_smi():
    """Check GPU status using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        print(f"✅ Found {len(lines)} GPU(s)\n")
        
        for line in lines[:8]:  # Limit to 8 GPUs
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 8:
                idx, name, total_mem, used_mem, free_mem, temp, util, power = parts
                print(f"GPU {idx}: {name}")
                print(f"  └─ Memory: {used_mem}/{total_mem} MB ({free_mem} MB free)")
                print(f"  └─ Temperature: {temp}°C")
                print(f"  └─ Utilization: {util}%")
                print(f"  └─ Power Draw: {power}W")
                print()
        
        return True
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ nvidia-smi error: {e}")
        return False

def main():
    print("=" * 50)
    print("GPU Status Check")
    print("=" * 50 + "\n")
    
    # Try PyTorch first (more detailed info)
    if not check_gpus_with_torch():
        # Fall back to nvidia-smi
        if not check_gpus_with_nvidia_smi():
            print("❌ No GPU detection method available")
            print("   Please install PyTorch or ensure nvidia-smi is in PATH")
            sys.exit(1)

if __name__ == "__main__":
    main()

