# zzChatTTS

A Python-based ChatTTS (Chat Text-To-Speech) implementation for converting text to natural-sounding speech.

## Features

- 🎙️ High-quality text-to-speech conversion using ChatTTS
- 🚀 Easy-to-use Python API
- 💻 Command-line interface for quick conversions
- 📦 Batch processing support
- 🎛️ Customizable voice parameters (temperature, top_p, top_k)
- 🔧 GPU acceleration support (CUDA)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
git clone https://github.com/Hell0zZ/zzChatTTS.git
cd zzChatTTS
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

### Python API

```python
from zzchattts import TTSEngine

# Initialize the TTS engine
tts = TTSEngine()

# Generate speech from text
audio = tts.generate_speech(
    text="Hello! Welcome to ChatTTS.",
    output_path="output.wav"
)
```

### Command Line Interface

```bash
# Basic usage
python examples/cli.py "Hello, this is a test" -o output.wav

# From a text file
python examples/cli.py input.txt -f -o output.wav

# With custom parameters
python examples/cli.py "Hello world" -o output.wav -t 0.5 -p 0.8 -k 30

# Force CPU usage
python examples/cli.py "Hello world" -o output.wav -d cpu
```

## Usage Examples

### Basic Example

```python
from zzchattts import TTSEngine

# Initialize engine
tts = TTSEngine()

# Simple text-to-speech
text = "This is a simple example of text-to-speech conversion."
audio = tts.generate_speech(text=text, output_path="output.wav")
```

### Batch Processing

```python
from zzchattts import TTSEngine

# Initialize engine
tts = TTSEngine()

# Convert multiple texts
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]

audio = tts.generate_speech(text=texts, output_path="batch_output.wav")
```

### Custom Parameters

```python
from zzchattts import TTSEngine

# Initialize engine
tts = TTSEngine(device='cuda')  # Use GPU

# Generate with custom parameters
audio = tts.generate_speech(
    text="Custom parameters example.",
    output_path="custom_output.wav",
    temperature=0.5,  # Higher = more random
    top_p=0.8,        # Nucleus sampling
    top_k=30          # Top-k sampling
)
```

## API Reference

### TTSEngine

Main class for text-to-speech conversion.

#### `__init__(device: Optional[str] = None)`

Initialize the TTS engine.

**Parameters:**
- `device` (str, optional): Device to use ('cuda', 'cpu', or None for auto-detection)

#### `generate_speech(text, output_path=None, temperature=0.3, top_p=0.7, top_k=20, sample_rate=24000)`

Generate speech from text.

**Parameters:**
- `text` (str or list): Input text or list of texts
- `output_path` (str, optional): Path to save the audio file
- `temperature` (float): Sampling temperature (default: 0.3)
- `top_p` (float): Nucleus sampling probability (default: 0.7)
- `top_k` (int): Top-k sampling parameter (default: 20)
- `sample_rate` (int): Output sample rate (default: 24000)

**Returns:**
- `numpy.ndarray`: Generated audio data

#### `save_audio(audio, output_path, sample_rate=24000)`

Save audio to file.

**Parameters:**
- `audio` (numpy.ndarray): Audio data
- `output_path` (str): Path to save the file
- `sample_rate` (int): Sample rate (default: 24000)

## Examples

The `examples/` directory contains several demonstration scripts:

- `basic_example.py`: Simple text-to-speech conversion
- `batch_example.py`: Batch processing multiple texts
- `cli.py`: Command-line interface

Run examples:

```bash
python examples/basic_example.py
python examples/batch_example.py
```

## Configuration

### Voice Parameters

- **Temperature** (0.0 - 1.0): Controls randomness in generation
  - Lower values (0.1-0.3): More consistent, stable output
  - Higher values (0.5-1.0): More varied, creative output

- **Top-P** (0.0 - 1.0): Nucleus sampling threshold
  - Controls diversity of word selection
  - Default: 0.7

- **Top-K** (int): Limits vocabulary to top K most likely words
  - Lower values: More focused output
  - Higher values: More diverse output
  - Default: 20

## Requirements

- ChatTTS >= 0.1.0
- torch >= 2.0.0
- torchaudio >= 2.0.0
- numpy >= 1.24.0
- soundfile >= 0.12.0

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on top of [ChatTTS](https://github.com/2noise/ChatTTS)
- Powered by PyTorch

## Support

For issues and questions, please open an issue on GitHub.
