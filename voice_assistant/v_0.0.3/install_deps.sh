#!/bin/bash
echo "Installing optimized dependencies for RTX 3090..."

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers with flash attention
pip install transformers accelerate
pip install flash-attn --no-build-isolation

# Install quantization support
pip install bitsandbytes

# Install audio libraries
pip install whisper openai-whisper
pip install TTS
pip install sounddevice pyaudio
pip install numpy scipy

# Install optional optimizations
pip install optimum
pip install auto-gptq  # For GPTQ models

echo "Installation complete!"
