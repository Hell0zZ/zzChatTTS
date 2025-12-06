"""
Unit tests for zzChatTTS
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch


class TestTTSEngine(unittest.TestCase):
    """Test cases for the TTSEngine class."""
    
    @patch('zzchattts.tts_engine.ChatTTS.Chat')
    @patch('zzchattts.tts_engine.torch')
    def setUp(self, mock_torch, mock_chat_class):
        """Set up test fixtures."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = False
        
        # Mock ChatTTS Chat instance
        self.mock_chat = Mock()
        mock_chat_class.return_value = self.mock_chat
        
        # Import after mocking
        from zzchattts import TTSEngine
        self.TTSEngine = TTSEngine
        
    @patch('zzchattts.tts_engine.ChatTTS.Chat')
    @patch('zzchattts.tts_engine.torch')
    def test_init_cpu(self, mock_torch, mock_chat_class):
        """Test initialization with CPU device."""
        mock_torch.cuda.is_available.return_value = False
        mock_chat = Mock()
        mock_chat_class.return_value = mock_chat
        
        from zzchattts import TTSEngine
        tts = TTSEngine()
        
        self.assertEqual(tts.device, 'cpu')
        mock_chat.load.assert_called_once()
    
    @patch('zzchattts.tts_engine.ChatTTS.Chat')
    @patch('zzchattts.tts_engine.torch')
    def test_init_cuda(self, mock_torch, mock_chat_class):
        """Test initialization with CUDA device."""
        mock_torch.cuda.is_available.return_value = True
        mock_chat = Mock()
        mock_chat_class.return_value = mock_chat
        
        from zzchattts import TTSEngine
        tts = TTSEngine()
        
        self.assertEqual(tts.device, 'cuda')
        mock_chat.load.assert_called_once()
    
    @patch('zzchattts.tts_engine.ChatTTS.Chat')
    @patch('zzchattts.tts_engine.torch')
    def test_generate_speech_single_text(self, mock_torch, mock_chat_class):
        """Test speech generation with a single text."""
        mock_torch.cuda.is_available.return_value = False
        mock_chat = Mock()
        mock_chat_class.return_value = mock_chat
        
        # Mock the inference output
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_chat.infer.return_value = [mock_audio]
        
        from zzchattts import TTSEngine
        tts = TTSEngine()
        
        result = tts.generate_speech("Hello world")
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, mock_audio)
        mock_chat.infer.assert_called_once()
    
    @patch('zzchattts.tts_engine.ChatTTS.Chat')
    @patch('zzchattts.tts_engine.torch')
    def test_generate_speech_multiple_texts(self, mock_torch, mock_chat_class):
        """Test speech generation with multiple texts."""
        mock_torch.cuda.is_available.return_value = False
        mock_chat = Mock()
        mock_chat_class.return_value = mock_chat
        
        # Mock the inference output
        mock_audio1 = np.array([0.1, 0.2])
        mock_audio2 = np.array([0.3, 0.4])
        mock_chat.infer.return_value = [mock_audio1, mock_audio2]
        
        from zzchattts import TTSEngine
        tts = TTSEngine()
        
        texts = ["Hello", "World"]
        result = tts.generate_speech(texts)
        
        self.assertIsInstance(result, np.ndarray)
        expected = np.concatenate([mock_audio1, mock_audio2])
        np.testing.assert_array_equal(result, expected)
    
    @patch('zzchattts.tts_engine.ChatTTS.Chat')
    @patch('zzchattts.tts_engine.torch')
    @patch('zzchattts.tts_engine.sf.write')
    def test_save_audio(self, mock_sf_write, mock_torch, mock_chat_class):
        """Test audio saving functionality."""
        mock_torch.cuda.is_available.return_value = False
        mock_chat = Mock()
        mock_chat_class.return_value = mock_chat
        
        from zzchattts import TTSEngine
        tts = TTSEngine()
        
        audio = np.array([0.1, 0.2, 0.3])
        output_path = "/tmp/test_output.wav"
        sample_rate = 24000
        
        tts.save_audio(audio, output_path, sample_rate)
        
        mock_sf_write.assert_called_once_with(output_path, audio, sample_rate)
    
    @patch('zzchattts.tts_engine.ChatTTS.Chat')
    @patch('zzchattts.tts_engine.torch')
    def test_generate_speech_with_parameters(self, mock_torch, mock_chat_class):
        """Test speech generation with custom parameters."""
        mock_torch.cuda.is_available.return_value = False
        mock_chat = Mock()
        mock_chat_class.return_value = mock_chat
        
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_chat.infer.return_value = [mock_audio]
        
        from zzchattts import TTSEngine
        tts = TTSEngine()
        
        result = tts.generate_speech(
            "Test",
            temperature=0.5,
            top_p=0.8,
            top_k=30
        )
        
        self.assertIsInstance(result, np.ndarray)
        mock_chat.infer.assert_called_once()


class TestImports(unittest.TestCase):
    """Test module imports."""
    
    def test_import_package(self):
        """Test that the package can be imported."""
        import zzchattts
        self.assertTrue(hasattr(zzchattts, 'TTSEngine'))
        self.assertTrue(hasattr(zzchattts, '__version__'))
    
    def test_version(self):
        """Test that version is defined."""
        import zzchattts
        self.assertIsInstance(zzchattts.__version__, str)
        self.assertEqual(zzchattts.__version__, '0.1.0')


if __name__ == '__main__':
    unittest.main()
