#!/usr/bin/env python3
"""
Example demonstrating batch text-to-speech conversion with multiple texts.
"""

from zzchattts import TTSEngine


def main():
    # Initialize the TTS engine
    print("Initializing ChatTTS engine...")
    tts = TTSEngine()
    
    # Multiple texts to convert
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "And this is the third sentence."
    ]
    
    # Generate speech for all texts
    print(f"\nConverting {len(texts)} texts to speech...")
    audio = tts.generate_speech(
        text=texts,
        output_path="batch_output.wav",
        temperature=0.3
    )
    
    print(f"\nGenerated audio shape: {audio.shape}")
    print("Done! Audio saved to 'batch_output.wav'")


if __name__ == "__main__":
    main()
