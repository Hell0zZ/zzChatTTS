#!/usr/bin/env python3
"""
Example usage of ChatTTS
"""

from chattts import load_model


def main():
    """Main example function."""
    print("=" * 50)
    print("ChatTTS Example")
    print("=" * 50)
    
    # Initialize the model
    print("\n1. Loading ChatTTS model...")
    model = load_model()
    
    # Example 1: Single text
    print("\n2. Generating speech from single text...")
    text = "Hello, this is a demonstration of ChatTTS."
    audio = model.generate(text)
    print(f"Generated audio shape: {audio.shape}")
    
    # Save the audio
    output_path = "output_single.wav"
    model.save_audio(audio, output_path)
    
    # Example 2: Multiple texts
    print("\n3. Generating speech from multiple texts...")
    texts = [
        "This is the first sentence.",
        "And this is the second sentence.",
        "Finally, this is the third sentence."
    ]
    audios = model.generate(texts)
    print(f"Generated {len(audios)} audio samples")
    
    # Save multiple audios
    for i, audio in enumerate(audios):
        output_path = f"output_{i}.wav"
        model.save_audio(audio, output_path)
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
