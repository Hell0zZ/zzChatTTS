# zzChatTTS Examples

This directory contains example scripts demonstrating how to use zzChatTTS.

## Examples

### basic_example.py
A simple example showing basic text-to-speech conversion.

```bash
python basic_example.py
```

### batch_example.py
Demonstrates batch processing of multiple texts.

```bash
python batch_example.py
```

### cli.py
A command-line interface for text-to-speech conversion.

```bash
# Basic usage
python cli.py "Your text here" -o output.wav

# From file
python cli.py input.txt -f -o output.wav

# With custom parameters
python cli.py "Hello" -o output.wav -t 0.5 -p 0.8 -k 30
```

## Requirements

Make sure you have installed all dependencies:

```bash
pip install -r ../requirements.txt
```

## Notes

- The first run will download the ChatTTS model, which may take some time
- Generated audio files will be saved in the current directory unless otherwise specified
- Use GPU (CUDA) for faster inference if available
