# transcription.py
import whisper
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model once when the module is imported.
# You can choose other models like "base", "medium", "large"
# "tiny" is fast but less accurate. "base" is a good starting point.
MODEL_SIZE = "base.en" # Using English-only model for efficiency
MODEL = None # Initialize MODEL as None

def load_whisper_model():
    """Loads the Whisper model if not already loaded."""
    global MODEL
    if MODEL is None:
        try:
            logger.info(f"Loading Whisper model: {MODEL_SIZE}...")
            MODEL = whisper.load_model(MODEL_SIZE)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}", exc_info=True) # Added exc_info for more details
            logger.error("Please ensure you have enough RAM and the model files are accessible.")
            logger.error("Try a smaller model like 'tiny.en' or 'base.en' if you have memory issues.")
            MODEL = None # Ensure it remains None if loading fails
    return MODEL


def transcribe_audio(audio_file_path):
    """
    Transcribes the given audio file path using the pre-loaded Whisper model.
    The model is loaded on the first call to transcribe_audio or if load_whisper_model() is called explicitly.
    """
    model_instance = load_whisper_model() # Ensures model is loaded
    
    if model_instance is None:
        logger.error("Whisper model not loaded. Cannot transcribe.")
        return None
        
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return None
    try:
        logger.info(f"Starting transcription for {audio_file_path}...")
        # For CPU, fp16 should be False. If you have a compatible GPU and CUDA setup, you might set it to True.
        result = model_instance.transcribe(audio_file_path, fp16=False) 
        logger.info(f"Transcription successful for {audio_file_path}.")
        return result["text"]
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # This part is for testing transcription.py directly
    # Ensure ffmpeg is installed and in your PATH
    
    # Pre-load the model for testing
    print("Attempting to load model for direct test...")
    load_whisper_model() 
    
    if MODEL: # Check if model loaded successfully
        test_audio_path = "test_audio.mp3" # <<<< IMPORTANT: REPLACE WITH A VALID PATH TO AN AUDIO FILE FOR TESTING >>>>
                                       # e.g., "C:/Users/varun/Music/sample.mp3" or a relative path if the file is in the same directory
        
        if os.path.exists(test_audio_path):
            print(f"Transcribing {test_audio_path}...")
            transcript = transcribe_audio(test_audio_path)
            if transcript:
                print("\nTranscript:")
                print(transcript)
            else:
                print("Transcription failed or returned empty.")
        else:
            print(f"Test audio file '{test_audio_path}' not found.")
            print("Please create a dummy audio file (e.g., test_audio.mp3) in the project root,")
            print("or update 'test_audio_path' in transcription.py to a valid audio file path to test this script directly.")
    else:
        print("Whisper model could not be loaded. Skipping direct transcription test.")