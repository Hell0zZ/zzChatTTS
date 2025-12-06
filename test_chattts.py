#!/usr/bin/env python3
"""
Simple tests for ChatTTS
"""

import unittest
import numpy as np
import os
from chattts import ChatTTS, load_model


class TestChatTTS(unittest.TestCase):
    """Test cases for ChatTTS functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = load_model(device='cpu')
        
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, ChatTTS)
        
    def test_single_text_generation(self):
        """Test generating audio from single text."""
        text = "Hello world"
        audio = self.model.generate(text)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        self.assertEqual(audio.dtype, np.float32)
        
    def test_batch_text_generation(self):
        """Test generating audio from multiple texts."""
        texts = ["First", "Second", "Third"]
        audios = self.model.generate(texts)
        
        self.assertIsInstance(audios, list)
        self.assertEqual(len(audios), 3)
        
        for audio in audios:
            self.assertIsInstance(audio, np.ndarray)
            self.assertGreater(len(audio), 0)
            
    def test_save_audio(self):
        """Test saving audio to file."""
        text = "Test audio"
        audio = self.model.generate(text)
        
        output_path = "/tmp/test_output.wav"
        self.model.save_audio(audio, output_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
            
    def test_generation_parameters(self):
        """Test that generation parameters are accepted."""
        text = "Test"
        audio = self.model.generate(
            text,
            temperature=0.8,
            top_p=0.95,
            top_k=100,
            max_length=1024
        )
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)


if __name__ == "__main__":
    unittest.main()
