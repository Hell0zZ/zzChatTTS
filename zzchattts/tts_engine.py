"""
ChatTTS Text-to-Speech Engine
"""

import ChatTTS
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Union, List


class TTSEngine:
    """
    A wrapper class for ChatTTS text-to-speech functionality.
    
    This class provides an easy-to-use interface for generating speech from text
    using the ChatTTS model.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the TTS engine.
        
        Args:
            device: Device to use for inference ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.chat = ChatTTS.Chat()
        self._load_model()
    
    def _load_model(self):
        """Load the ChatTTS model."""
        print(f"Loading ChatTTS model on {self.device}...")
        self.chat.load(compile=False, device=self.device)
        print("Model loaded successfully!")
    
    def generate_speech(
        self,
        text: Union[str, List[str]],
        output_path: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
        sample_rate: int = 24000
    ) -> np.ndarray:
        """
        Generate speech from text.
        
        Args:
            text: Input text or list of texts to convert to speech
            output_path: Optional path to save the output audio file
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            sample_rate: Output audio sample rate
            
        Returns:
            Generated audio as numpy array
        """
        # Convert single string to list
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Configure inference parameters
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            temperature=temperature,
            top_P=top_p,
            top_K=top_k,
        )
        
        params_refine_text = ChatTTS.Chat.RefineTextParams()
        
        print(f"Generating speech for {len(texts)} text(s)...")
        
        # Generate speech
        wavs = self.chat.infer(
            texts,
            params_infer_code=params_infer_code,
            params_refine_text=params_refine_text,
        )
        
        # Convert to numpy array
        if len(wavs) == 1:
            audio = wavs[0]
        else:
            # Concatenate multiple outputs
            audio = np.concatenate(wavs, axis=0)
        
        # Save to file if output path is provided
        if output_path:
            self.save_audio(audio, output_path, sample_rate)
            print(f"Audio saved to {output_path}")
        
        return audio
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: int = 24000
    ):
        """
        Save audio to file.
        
        Args:
            audio: Audio data as numpy array
            output_path: Path to save the audio file
            sample_rate: Sample rate of the audio
        """
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using soundfile
        sf.write(output_path, audio, sample_rate)
    
    def generate_with_speaker(
        self,
        text: Union[str, List[str]],
        speaker_embedding: Optional[torch.Tensor] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate speech with a specific speaker voice.
        
        Args:
            text: Input text or list of texts
            speaker_embedding: Speaker embedding tensor (None for random)
            output_path: Optional path to save the output
            **kwargs: Additional parameters for generate_speech
            
        Returns:
            Generated audio as numpy array
        """
        # This is a placeholder for speaker-specific generation
        # The actual implementation would depend on ChatTTS's speaker control features
        return self.generate_speech(text, output_path, **kwargs)
