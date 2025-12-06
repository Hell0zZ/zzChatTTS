#!/usr/bin/env python3
"""
Simple example demonstrating basic text-to-speech conversion using zzChatTTS.
"""

from zzchattts import TTSEngine


def main():
    # Initialize the TTS engine
    print("Initializing ChatTTS engine...")
    tts = TTSEngine()
    
    # Example text
    text = "Hello! Welcome to ChatTTS. This is a text-to-speech conversion demo."
    
    # Generate speech and save to file
    print(f"\nConverting text to speech: '{text}'")
    audio = tts.generate_speech(
        text=text,
        output_path="output.wav",
        temperature=0.3,
        top_p=0.7,
        top_k=20
    )
    
    print(f"\nGenerated audio shape: {audio.shape}")
    print("Done! Audio saved to 'output.wav'")


if __name__ == "__main__":
    main()
