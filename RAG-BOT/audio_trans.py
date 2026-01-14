import os
import io
import wave
import pyaudio
import threading
import time
from pathlib import Path
from agno.agent import Agent
from agno.media import Audio
from agno.models.google import Gemini

class TeluguToEnglishVoiceTranslator:
    def __init__(self):
        # Initialize the Gemini agent for transcription and translation
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key="AIzaSyBWlCMq91iKL0_XhxQhPU2MlKAr67lXLiM"),
            description="You are an expert in Telugu language transcription and English translation.",
            instructions=[
                "When given Telugu audio, first transcribe it accurately to Telugu text.",
                "Then translate the Telugu text to clear, natural English.",
                "Provide both the Telugu transcription and English translation.",
                "If the audio is unclear, mention that in your response."
            ],
            markdown=True,
        )
        
        # Audio recording parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.recording = False
        self.frames = []
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.recording = True
        self.frames = []
        
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("🎤 Recording started... Press Enter to stop recording")
        
        while self.recording:
            data = stream.read(self.chunk)
            self.frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        print("🛑 Recording stopped")
        
    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False
        
    def save_audio_to_bytes(self):
        """Convert recorded frames to audio bytes"""
        if not self.frames:
            return None
            
        # Create a BytesIO object to store the WAV data
        audio_buffer = io.BytesIO()
        
        # Create WAV file in memory
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
        
        # Get the bytes
        audio_buffer.seek(0)
        return audio_buffer.read()
    
    def transcribe_and_translate(self, audio_bytes):
        """Transcribe Telugu audio and translate to English using Gemini"""
        try:
            response = self.agent.run(
                "Please transcribe this Telugu audio and then translate it to English. "
                "Provide both the Telugu transcription and English translation clearly.",
                audio=[Audio(content=audio_bytes)]
            )
            return response.content
        except Exception as e:
            return f"Error during transcription/translation: {str(e)}"
    
    def record_and_translate(self):
        """Main method to record audio and get translation"""
        # Start recording in a separate thread
        recording_thread = threading.Thread(target=self.start_recording)
        recording_thread.start()
        
        # Wait for user to press Enter to stop recording
        input()
        self.stop_recording()
        recording_thread.join()
        
        # Convert to audio bytes
        audio_bytes = self.save_audio_to_bytes()
        
        if audio_bytes:
            print("\n🔄 Processing audio...")
            result = self.transcribe_and_translate(audio_bytes)
            print("\n📝 Result:")
            print("=" * 50)
            print(result)
            print("=" * 50)
        else:
            print("❌ No audio recorded")

# Alternative: File-based approach for existing audio files
class FileBasedTranslator:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            description="You are an expert in Telugu language transcription and English translation.",
            instructions=[
                "When given Telugu audio, first transcribe it accurately to Telugu text.",
                "Then translate the Telugu text to clear, natural English.",
                "Provide both the Telugu transcription and English translation.",
                "Format your response clearly with sections for transcription and translation."
            ],
            markdown=True,
        )
    
    def translate_from_file(self, audio_file_path):
        """Translate Telugu audio file to English"""
        try:
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                return f"Audio file not found: {audio_file_path}"
            
            response = self.agent.run(
                "Please transcribe this Telugu audio and translate it to English. "
                "Provide both the Telugu transcription and English translation.",
                audio=[Audio(filepath=audio_path)]
            )
            return response.content
        except Exception as e:
            return f"Error processing file: {str(e)}"
    
    def translate_from_url(self, audio_url):
        """Translate Telugu audio from URL to English"""
        try:
            import requests
            response = requests.get(audio_url)
            response.raise_for_status()
            audio_content = response.content
            
            result = self.agent.run(
                "Please transcribe this Telugu audio and translate it to English. "
                "Provide both the Telugu transcription and English translation.",
                audio=[Audio(content=audio_content)]
            )
            return result.content
        except Exception as e:
            return f"Error processing URL: {str(e)}"

# Usage examples
def main():
    # Set your Google API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ Please set your GOOGLE_API_KEY environment variable")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    print("Telugu to English Voice Translator")
    print("=" * 40)
    print("1. Record live audio")
    print("2. Translate from file")
    print("3. Translate from URL")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        # Live recording
        translator = TeluguToEnglishVoiceTranslator()
        translator.record_and_translate()
        
    elif choice == "2":
        # File-based translation
        file_path = input("Enter audio file path: ").strip()
        translator = FileBasedTranslator()
        result = translator.translate_from_file(file_path)
        print("\n📝 Translation Result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
    elif choice == "3":
        # URL-based translation
        audio_url = input("Enter audio URL: ").strip()
        translator = FileBasedTranslator()
        result = translator.translate_from_url(audio_url)
        print("\n📝 Translation Result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()