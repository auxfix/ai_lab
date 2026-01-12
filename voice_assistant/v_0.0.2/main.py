import io
import queue
import threading
import time
import wave

import numpy as np
import pyaudio
import sounddevice as sd
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Coqui TTS
try:
    from TTS.api import TTS

    COQUI_AVAILABLE = True
except ImportError:
    print("Coqui TTS not installed. Install with: pip install TTS")
    COQUI_AVAILABLE = False
    from TTS.api import TTS  # This will fail if not installed


class VoiceCompanion:
    def __init__(
        self, tts_engine="coqui", coqui_model="tts_models/en/ljspeech/tacotron2-DDC"
    ):
        with torch.no_grad():
            torch.cuda.empty_cache()

        # Load Whisper model
        self.whisper_model = whisper.load_model("base")

        # Load GPT-NeoX-20B (requires significant VRAM)
        print("Loading gpt-oss-20B...")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b", dtype=torch.float16, device_map="auto"
        )

        # Initialize TTS engine
        self.tts_engine = tts_engine
        self.coqui_model_name = coqui_model

        if tts_engine == "coqui" and COQUI_AVAILABLE:
            print(f"Loading Coqui TTS model: {coqui_model}")
            # Initialize Coqui TTS
            self.tts = TTS(model_name=coqui_model, progress_bar=False)

            # Check available models
            # print("Available Coqui TTS models:", TTS().list_models())

            # Initialize PyAudio for playback
            self.p = pyaudio.PyAudio()

            # Default voice settings (can be changed)
            self.voice_settings = {
                "speaker": None,  # For multi-speaker models
                "emotion": None,  # Emotion for emotional TTS models
                "speed": 1.0,  # Speech speed
            }
        else:
            print("Falling back to pyttsx3")
            import pyttsx3

            self.tts_engine = "pyttsx3"
            self.tts = pyttsx3.init()

        # Conversation history
        self.conversation_history = []

        # Audio recording settings
        self.sample_rate = 16000
        self.channels = 1
        self.audio_queue = queue.Queue()
        self.is_recording = False

    def transcribe_audio(self, audio_data):
        """Convert speech to text using Whisper"""
        result = self.whisper_model.transcribe(audio_data)
        return result["text"]

    def generate_response(self, user_input):
        """Generate response using GPT-NeoX-20B"""
        # Format conversation context
        prompt = (
            "\n".join(self.conversation_history[-5:])
            + f"\nUser: {user_input}\nAssistant:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()

        # Update conversation history
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")

        return response

    def speak_with_coqui(self, text, output_path=None, play_audio=True):
        """Convert text to speech using Coqui TTS"""
        try:
            if output_path is None:
                # Generate audio in memory
                wav_data = self.tts.tts(
                    text=text,
                    speaker=self.voice_settings["speaker"],
                    language=self.voice_settings["language"],
                    emotion=self.voice_settings["emotion"],
                    speed=self.voice_settings["speed"],
                )

                # Convert to numpy array if needed
                if isinstance(wav_data, list):
                    wav_data = np.array(wav_data)

                if play_audio:
                    self._play_audio(wav_data, self.tts.synthesizer.output_sample_rate)

                return wav_data
            else:
                # Save to file
                self.tts.tts_to_file(
                    text=text,
                    speaker=self.voice_settings["speaker"],
                    language=self.voice_settings["language"],
                    emotion=self.voice_settings["emotion"],
                    speed=self.voice_settings["speed"],
                    file_path=output_path,
                )

                if play_audio:
                    # Load and play the saved file
                    self._play_audio_file(output_path)

                return output_path

        except Exception as e:
            print(f"Coqui TTS error: {e}")
            # Fallback to pyttsx3 if available
            if self.tts_engine == "pyttsx3":
                self.speak_with_pyttsx3(text)
            return None

    def _play_audio(self, audio_data, sample_rate=22050):
        """Play audio data using PyAudio"""
        stream = self.p.open(
            format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True
        )

        # Ensure audio data is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        stream.write(audio_data.tobytes())
        stream.stop_stream()
        stream.close()

    def _play_audio_file(self, file_path):
        """Play audio from file"""
        import wave

        wf = wave.open(file_path, "rb")

        stream = self.p.open(
            format=self.p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()
        wf.close()

    def speak_with_pyttsx3(self, text):
        """Fallback to pyttsx3"""
        self.tts.say(text)
        self.tts.runAndWait()

    def speak(self, text, output_file=None, play_audio=True):
        """Convert text to speech using selected engine"""
        if self.tts_engine == "coqui" and COQUI_AVAILABLE:
            return self.speak_with_coqui(text, output_file, play_audio)
        else:
            self.speak_with_pyttsx3(text)
            return None

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds...")

        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()

        return audio_data.flatten()

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for streaming audio"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def start_streaming_recording(self):
        """Start streaming audio recording"""
        self.is_recording = True
        self.audio_queue = queue.Queue()

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
        )
        stream.start()
        return stream

    def stop_streaming_recording(self, stream):
        """Stop streaming audio recording"""
        self.is_recording = False
        stream.stop()
        stream.close()

        # Collect all audio data
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())

        if audio_chunks:
            return np.concatenate(audio_chunks)
        return np.array([])

    def run_conversation(self, use_mic=False):
        """Main conversation loop"""
        print("Voice Companion Ready! Press Ctrl+C to exit.")
        print(f"Using TTS engine: {self.tts_engine}")

        if use_mic and self.tts_engine == "coqui":
            print("Voice settings:", self.voice_settings)

        while True:
            if use_mic:
                # Record audio from microphone
                print("\nListening... (press Enter when done)")
                input("Press Enter to start recording...")

                audio_data = self.record_audio(duration=5)

                # Save audio for debugging
                # import scipy.io.wavfile as wav
                # wav.write("temp_audio.wav", self.sample_rate, audio_data)

                # Transcribe
                user_text = self.transcribe_audio(audio_data)
                print(f"You said: {user_text}")
            else:
                # Text input mode
                user_text = input("\nYou: ")

            if user_text.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            if not user_text.strip():
                print("No input detected. Try again.")
                continue

            # Generate response
            print("Thinking...")
            response = self.generate_response(user_text)
            print(f"Assistant: {response}")

            # Speak response
            print("Speaking...")
            self.speak(response)

            # Optional: Save audio to file
            # self.speak(response, output_file="response.wav")

    def set_voice_settings(self, speaker=None, language="en", emotion=None, speed=1.0):
        """Update Coqui TTS voice settings"""
        self.voice_settings = {
            "speaker": speaker,
            "language": language,
            "emotion": emotion,
            "speed": speed,
        }

    def list_coqui_models(self):
        """List available Coqui TTS models"""
        if COQUI_AVAILABLE:
            models = TTS().list_models()
            print("\nAvailable Coqui TTS models:")
            for model in models:
                print(f"  - {model}")
            return models
        return []

    def __del__(self):
        """Cleanup"""
        if hasattr(self, "p") and self.tts_engine == "coqui":
            self.p.terminate()


if __name__ == "__main__":
    # Initialize with Coqui TTS
    companion = VoiceCompanion(
        tts_engine="coqui",
        # Try different models:
        # coqui_model="tts_models/en/ljspeech/tacotron2-DDC"  # Default
        # coqui_model="tts_models/en/vctk/vits"  # Multi-speaker
        coqui_model="tts_models/en/jenny/jenny"  # Female voice
    )

    # Optional: List available models
    companion.list_coqui_models()

    # Optional: Change voice settings (for multi-speaker models)
    # companion.set_voice_settings(speaker="p225", speed=1.1)

    # Run conversation
    # Set use_mic=True to use microphone input
    companion.run_conversation(use_mic=True)
