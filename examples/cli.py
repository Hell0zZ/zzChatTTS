#!/usr/bin/env python3
"""
Command-line interface for ChatTTS text-to-speech conversion.
"""

import argparse
from pathlib import Path
from zzchattts import TTSEngine


def main():
    parser = argparse.ArgumentParser(
        description="ChatTTS - Convert text to speech using ChatTTS"
    )
    
    parser.add_argument(
        "text",
        type=str,
        help="Text to convert to speech (or path to text file if --file is used)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    
    parser.add_argument(
        "-f", "--file",
        action="store_true",
        help="Treat input as a file path and read text from file"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)"
    )
    
    parser.add_argument(
        "-p", "--top_p",
        type=float,
        default=0.7,
        help="Top-p sampling parameter (default: 0.7)"
    )
    
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=20,
        help="Top-k sampling parameter (default: 20)"
    )
    
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use for inference (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Read text from file if --file flag is used
    if args.file:
        text_path = Path(args.text)
        if not text_path.exists():
            print(f"Error: File '{args.text}' not found")
            return 1
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"Read text from file: {args.text}")
    else:
        text = args.text
    
    # Initialize TTS engine
    print("Initializing ChatTTS engine...")
    tts = TTSEngine(device=args.device)
    
    # Generate speech
    print(f"\nConverting text to speech...")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    audio = tts.generate_speech(
        text=text,
        output_path=args.output,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    print(f"\nSuccess! Generated audio shape: {audio.shape}")
    print(f"Audio saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
