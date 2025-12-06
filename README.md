# zzChatTTS

A Text-to-Speech (TTS) implementation using ChatTTS technology.

## Features

- Simple and easy-to-use API
- Support for single and batch text processing
- Configurable generation parameters
- Audio output in WAV format
- GPU acceleration support (when available)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hell0zZ/zzChatTTS.git
cd zzChatTTS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from chattts import load_model

# Load the model
model = load_model()

# Generate speech from text
text = "Hello, this is ChatTTS!"
audio = model.generate(text)

# Save the audio
model.save_audio(audio, "output.wav")
```

### Batch Processing

```python
from chattts import load_model

# Load the model
model = load_model()

# Generate speech from multiple texts
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]
audios = model.generate(texts)

# Save all audios
for i, audio in enumerate(audios):
    model.save_audio(audio, f"output_{i}.wav")
```

### Advanced Options

```python
from chattts import load_model

model = load_model(device='cuda')  # Use GPU

audio = model.generate(
    text="Your text here",
    temperature=0.8,  # Higher = more random
    top_p=0.9,        # Nucleus sampling
    top_k=50          # Top-k sampling
)
```

## Running the Example

```bash
python example.py
```

This will generate several example audio files demonstrating the capabilities of ChatTTS.

## API Reference

### ChatTTS Class

#### `__init__(device=None)`
Initialize the ChatTTS model.

**Parameters:**
- `device` (str, optional): Device to run on ('cuda', 'cpu', or None for auto-detect)

#### `generate(text, temperature=0.7, top_p=0.9, top_k=50, max_length=2048)`
Generate speech from text.

**Parameters:**
- `text` (str or list): Input text(s) to convert to speech
- `temperature` (float): Sampling temperature (default: 0.7)
- `top_p` (float): Nucleus sampling threshold (default: 0.9)
- `top_k` (int): Top-k sampling threshold (default: 50)
- `max_length` (int): Maximum sequence length (default: 2048)

**Returns:**
- Audio waveform(s) as numpy array(s)

#### `save_audio(audio, path, sample_rate=22050)`
Save audio to a WAV file.

**Parameters:**
- `audio` (np.ndarray): Audio waveform
- `path` (str): Output file path
- `sample_rate` (int): Sample rate in Hz (default: 22050)

### Helper Functions

#### `load_model(device=None)`
Convenience function to load a ChatTTS model.

**Parameters:**
- `device` (str, optional): Device to run on

**Returns:**
- ChatTTS instance

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- SciPy 1.10+

See `requirements.txt` for full dependency list.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
