#!/usr/bin/env python3
"""
Environment and hardware verification utility for Nano-Voice.
Checks Python version, PyTorch compute devices (CUDA / MPS / CPU), and audio input/output devices.
"""

import sys
import platform

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_system():
    print_header("System & Python Environment")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {platform.python_version()} ({sys.executable})")

def check_pytorch():
    print_header("PyTorch & Accelerator Check")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check CUDA
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            print(f"  Device Count: {device_count}")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"  [{i}] {name} (Compute Cap: {cap[0]}.{cap[1]}, Total VRAM: {mem:.2f} GB)")
                
        # Check Apple Silicon MPS
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"Apple MPS Available: {mps_avail}")
        
        # Selected default compute device
        if cuda_avail:
            recommended = "cuda"
        elif mps_avail:
            recommended = "mps"
        else:
            recommended = "cpu"
        print(f"Recommended Accelerator: '{recommended}'")
        
    except ImportError:
        print("[!] PyTorch is NOT installed in this environment.")
        return False
    return True

def check_audio():
    print_header("Audio I/O Device Check")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in, default_out = sd.default.device
        print(f"Default Input Device ID: {default_in}")
        print(f"Default Output Device ID: {default_out}\n")
        
        print("Available Devices:")
        for idx, dev in enumerate(devices):
            in_ch = dev.get("max_input_channels", 0)
            out_ch = dev.get("max_output_channels", 0)
            kind = []
            if in_ch > 0:
                kind.append(f"IN:{in_ch}ch")
            if out_ch > 0:
                kind.append(f"OUT:{out_ch}ch")
            kind_str = ", ".join(kind) if kind else "Disabled"
            marker = ""
            if idx == default_in and idx == default_out:
                marker = " [DEFAULT IN/OUT]"
            elif idx == default_in:
                marker = " [DEFAULT IN]"
            elif idx == default_out:
                marker = " [DEFAULT OUT]"
            print(f"  [{idx:2d}] {dev['name']} ({kind_str}){marker}")
            
    except ImportError:
        print("[!] sounddevice is NOT installed or PortAudio is missing.")
    except Exception as e:
        print(f"[!] Warning while querying audio devices: {e}")

def check_optional_packages():
    print_header("Auxiliary Modules Check")
    modules = [
        ("faster_whisper", "Faster-Whisper (STT)"),
        ("piper", "Piper-TTS (Speech Synthesis)"),
        ("faiss", "FAISS (Vector Search)"),
        ("tiktoken", "Tiktoken (BPE Tokenization)"),
    ]
    for mod_name, label in modules:
        try:
            __import__(mod_name)
            print(f"  [✓] {label}: Available")
        except ImportError:
            print(f"  [-] {label}: Not installed (optional for training core, needed for edge voice)")

def main():
    print("=" * 60)
    print("        NANO-VOICE EDGE ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    check_system()
    check_pytorch()
    check_audio()
    check_optional_packages()
    print("\nEnvironment check complete.\n")

if __name__ == "__main__":
    main()
