# NVIDIA Jetson Orin Nano Deployment & Optimization Guide

This guide explains how to set up, configure, and optimize an **NVIDIA Jetson Orin Nano (8GB)** to run the **Nano-Voice** real-time voice assistant.

---

## 1. Prerequisites

- **Board**: NVIDIA Jetson Orin Nano Developer Kit (8GB)
- **OS**: NVIDIA JetPack 5.1.x or JetPack 6.x (Ubuntu 20.04 / 22.04 LTS)
- **Storage**: Fast NVMe SSD (M.2 Key M) strongly recommended over microSD
- **Peripherals**:
  - USB Conference Microphone (e.g. Jabra, Anker, or basic USB condenser mic)
  - 3.5mm audio jack speaker or USB audio speaker

---

## 2. Power Mode & Clock Optimization

By default, Jetson Orin Nano may run in low-power mode (7W). For maximum inference throughput:

```bash
# 1. Set power mode to 15W MAXN (all 6 ARM cores and GPU active)
sudo nvpmodel -m 0

# 2. Lock GPU and CPU clocks to maximum frequency
sudo jetson_clocks
```

---

## 3. Memory & Swap Configuration

Unified memory is shared between the CPU and GPU. To ensure stability and prevent Out-Of-Memory (OOM) kills during model loading:

```bash
# Create an 8GB swapfile on NVMe SSD
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Headless Mode (Optional - Saves ~1.2 GB RAM)
If using the Jetson primarily as a voice device without an attached monitor:
```bash
# Disable Ubuntu Desktop GUI
sudo systemctl set-default multi-user.target
sudo reboot

# To re-enable desktop GUI later if desired:
# sudo systemctl set-default graphical.target
```

---

## 4. Audio Device Setup & Verification

Connect your USB microphone and speaker, then run:

```bash
# Check ALSA audio recording devices
arecord -l

# Check ALSA audio playback devices
aplay -l
```

Run the built-in diagnostic utility to verify devices:
```bash
python3 tools/check_env.py
```

The script will list all detected input and output device IDs. Note your microphone device ID (e.g., `device_id = 2`).

---

## 5. Software Installation

```bash
# Install system audio libraries
sudo apt-get update && sudo apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

---

## 6. Launching Nano-Voice on Jetson

### Option A: Using Your Custom From-Scratch Model
```bash
# 1. Run 30s toy training or copy your trained model_final.pt to checkpoints/
python3 tools/train_toy.py

# 2. Start voice assistant
python3 run_voice.py --model scratch
```

### Option B: Using Instruct Model (SmolLM2 / Qwen)
```bash
python3 run_voice.py --model instruct
```

### Option C: Text-Only Testing Mode (No Mic Required)
```bash
python3 run_voice.py --model scratch --text-only
```
