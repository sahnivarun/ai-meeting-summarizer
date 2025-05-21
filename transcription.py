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


# # transcription.py
# import whisper
# import os
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
# logger = logging.getLogger(__name__)

# # --- Model Configuration ---
# # Option 1: Small and Fast (current)
# # MODEL_SIZE = "base.en"

# # Option 2: Medium - Better accuracy, slower, more RAM
# MODEL_SIZE = "medium.en"

# # Option 3: Large - Best accuracy, slowest, most RAM/VRAM needed
# # Consider large-v2 or large-v3 if your openai-whisper version supports them and you have the resources
# # MODEL_SIZE = "large-v2" # or "large-v3" or just "large"
# # --- End Model Configuration ---

# MODEL = None # Initialize MODEL as None

# def load_whisper_model():
#     """Loads the Whisper model if not already loaded."""
#     global MODEL
#     if MODEL is None:
#         try:
#             logger.info(f"Attempting to load Whisper model: {MODEL_SIZE}...")
#             MODEL = whisper.load_model(MODEL_SIZE)
#             logger.info(f"Whisper model '{MODEL_SIZE}' loaded successfully.")
#         except Exception as e:
#             logger.error(f"Error loading Whisper model '{MODEL_SIZE}': {e}", exc_info=True)
#             logger.error("Please ensure you have enough RAM/VRAM and the model files are accessible/downloadable.")
#             logger.error(f"If '{MODEL_SIZE}' is too large, try a smaller one like 'base.en' or 'small.en'.")
#             MODEL = None # Ensure it remains None if loading fails
#     return MODEL


# def transcribe_audio(audio_file_path: str) -> str | None:
#     """
#     Transcribes the given audio file path using the pre-loaded Whisper model.
#     """
#     model_instance = load_whisper_model() # Ensures model is loaded
    
#     if model_instance is None:
#         logger.error("Whisper model not loaded. Cannot transcribe.")
#         return None
        
#     if not os.path.exists(audio_file_path):
#         logger.error(f"Audio file not found: {audio_file_path}")
#         return None
    
#     try:
#         logger.info(f"Starting transcription for '{audio_file_path}' using model '{MODEL_SIZE}'...")
        
#         # --- Transcription Options ---
#         # You can experiment with these. For now, we'll keep it simple.
#         # See Whisper documentation for more details: https://github.com/openai/whisper
#         transcribe_options = dict(
#             fp16=False, # Set to True if using GPU and have CUDA setup for mixed precision (can be faster)
#             # language="en", # Explicitly set language if known, though ".en" models are English-only
#             # word_timestamps=True, # Set to True to get word-level timestamps (might influence VAD)
#             # vad_filter=True, # Enable VAD filtering (requires word_timestamps=True usually)
#             # no_speech_threshold=0.5, # Lower might transcribe more quiet parts (default is often 0.6)
#             # logprob_threshold=None, # Set to a value like -1.0 to be less strict, None for default
#             # beam_size=5, # Default is 5, larger might be more accurate but slower
#             # condition_on_previous_text=True, # Default, helps with context for long audio
#         )
#         # Remove options not needed to keep it clean:
#         if not transcribe_options.get("word_timestamps", False) and "vad_filter" in transcribe_options:
#             del transcribe_options["vad_filter"] # vad_filter usually needs word_timestamps

#         logger.info(f"Transcription options: {transcribe_options}")

#         result = model_instance.transcribe(audio_file_path, **transcribe_options)
        
#         transcribed_text = result.get("text", "")
#         if transcribed_text:
#             logger.info(f"Transcription successful for '{audio_file_path}'. Text length: {len(transcribed_text)} chars.")
#             # logger.debug(f"Full transcript for '{audio_file_path}':\n{transcribed_text}") # Uncomment for verbose logging
#         else:
#             logger.warning(f"Transcription for '{audio_file_path}' resulted in empty text. Segments found: {len(result.get('segments', []))}")

#         return transcribed_text

#     except Exception as e:
#         logger.error(f"Error during transcription for '{audio_file_path}': {e}", exc_info=True)
#         return None

# if __name__ == '__main__':
#     # This part is for testing transcription.py directly
#     # Ensure ffmpeg is installed and in your PATH
    
#     print("--- Direct Test of transcription.py ---")
#     # Pre-load the model for testing
#     print(f"Attempting to load model '{MODEL_SIZE}' for direct test...")
#     load_whisper_model() 
    
#     if MODEL: # Check if model loaded successfully
#         # <<<< IMPORTANT: REPLACE WITH A VALID PATH TO YOUR LONG SONG AUDIO FILE FOR TESTING >>>>
#         test_audio_path = "YOUR_LONG_SONG_AUDIO_FILE.mp3" 
#         # Example: test_audio_path = "C:/Users/varun/Music/MyLongSong.mp3"
        
#         if os.path.exists(test_audio_path):
#             print(f"\nTranscribing '{test_audio_path}' using model '{MODEL_SIZE}'...")
#             transcript = transcribe_audio(test_audio_path)
#             if transcript:
#                 print("\n--- Transcript ---")
#                 print(transcript)
#                 print(f"\n--- End of Transcript (Length: {len(transcript)} chars) ---")
#             else:
#                 print("\nTranscription failed or returned empty text.")
#         else:
#             if test_audio_path == "YOUR_LONG_SONG_AUDIO_FILE.mp3":
#                 print(f"\nSKIPPING TRANSCRIPTION TEST: Please update 'test_audio_path' in the "
#                       f"'if __name__ == \"__main__\":' block of transcription.py "
#                       f"to a valid audio file path to test this script directly.")
#             else:
#                 print(f"\nTest audio file '{test_audio_path}' not found.")
#     else:
#         print(f"\nWhisper model '{MODEL_SIZE}' could not be loaded. Skipping direct transcription test.")
#     print("--- End of Direct Test ---")