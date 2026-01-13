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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import Coqui TTS
try:
    from TTS.api import TTS

    COQUI_AVAILABLE = True
except ImportError:
    print("Coqui TTS not installed. Install with: pip install TTS")
    COQUI_AVAILABLE = False


class VoiceCompanion:
    def __init__(
        self,
        tts_engine="coqui",
        coqui_model="tts_models/en/ljspeech/tacotron2-DDC",
        llm_model="microsoft/phi-2",  # Change to your preferred model
        use_4bit=True,
        use_8bit=False,
    ):
        print(f"Initializing Voice Companion on RTX 3090 (24GB VRAM)...")

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Optimize memory allocation
        torch.backends.cuda.max_split_size_mb = 1024  # Larger split size for 3090
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU

        # Load Whisper model on GPU
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(
            "medium"
        ).cuda()  # Use medium for better accuracy

        # Configure quantization based on model size
        print(f"Loading LLM: {llm_model}")

        if "20b" in llm_model.lower() or "20B" in llm_model:
            # For 20B models, use 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            print("Using 4-bit quantization for 20B model")
        elif use_4bit:
            # 4-bit for smaller models
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization")
        elif use_8bit:
            # 8-bit quantization
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Using 8-bit quantization")
        else:
            quantization_config = None
            print("Using full precision (16-bit)")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model, trust_remote_code=True, padding_side="left"
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not quantization_config else None,
            max_memory={0: "22GB", "cpu": "64GB"} if not quantization_config else None,
            offload_folder="./offload" if "20b" in llm_model.lower() else None,
            low_cpu_mem_usage=True,
        )

        print(f"Model loaded on device: {self.model.device}")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # Initialize TTS engine
        self.tts_engine = tts_engine
        self.coqui_model_name = coqui_model

        if tts_engine == "coqui" and COQUI_AVAILABLE:
            print(f"Loading Coqui TTS model: {coqui_model}")
            self.tts = TTS(
                model_name=coqui_model, progress_bar=False, gpu=True
            )  # Use GPU

            # Initialize PyAudio for playback
            self.p = pyaudio.PyAudio()

            # Voice settings
            self.voice_settings = {
                "speaker": None,
                "language": "en",
                "emotion": None,
                "speed": 1.0,
            }
        else:
            print("Falling back to pyttsx3")
            import pyttsx3

            self.tts_engine = "pyttsx3"
            self.tts = pyttsx3.init()

        # Conversation history
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges

        # Audio recording settings
        self.sample_rate = 16000
        self.channels = 1
        self.audio_queue = queue.Queue()
        self.is_recording = False

        # Performance tracking
        self.response_times = []

    def transcribe_audio(self, audio_data):
        """Convert speech to text using Whisper on GPU"""
        # Ensure audio is in correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize audio
        audio_data = (
            audio_data / np.max(np.abs(audio_data))
            if np.max(np.abs(audio_data)) > 0
            else audio_data
        )

        # Transcribe
        result = self.whisper_model.transcribe(
            audio_data,
            language="en",
            fp16=True,  # Use FP16 for speed
            temperature=0.0,  # More deterministic
            best_of=5,
            beam_size=5,
        )
        return result["text"].strip()

    def generate_response(self, user_input):
        """Generate response using LLM with optimized settings"""
        start_time = time.time()

        # Format conversation context
        context = "\n".join(self.conversation_history[-self.max_history :])
        prompt = f"{context}\nUser: {user_input}\nAssistant:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Adjust based on model
        ).to(self.model.device)

        # Generate with optimized settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,  # Shorter for voice
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                num_beams=3,  # Beam search for better quality
                early_stopping=True,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()

        # Clean up response
        response = response.split("\nUser:")[0].split("\n\n")[0]

        # Update conversation history
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")

        # Track performance
        elapsed = time.time() - start_time
        self.response_times.append(elapsed)

        return response

    def speak_with_coqui(self, text):
        """Convert text to speech using Coqui TTS - optimized version"""

        self.tts.tts_to_file(
            text=text,
            speaker_wav="my/cloning/audio.wav",
            language="en",
            file_path="~/anser.wav",
        )

    def _play_audio(self, audio_data, sample_rate=22050):
        """Play audio data using PyAudio"""
        stream = self.p.open(
            format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True
        )

        # Ensure audio data is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        stream.write(audio_data.tobytes())
        stream.stop_stream()
        stream.close()

    def speak_with_pyttsx3(self, text):
        """Fallback to pyttsx3"""
        self.tts.say(text)
        self.tts.runAndWait()

    def speak(self, text):
        """Convert text to speech using selected engine"""
        if self.tts_engine == "coqui" and COQUI_AVAILABLE:
            return self.speak_with_coqui(text)

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds...")

        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=None,  # Use default device
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
            device=None,
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

    def run_conversation(self, use_mic=False, continuous=False):
        """Main conversation loop"""
        print("\n" + "=" * 50)
        print("Voice Companion Ready!")
        print(f"Using TTS engine: {self.tts_engine}")
        print("=" * 50 + "\n")

        if continuous and use_mic:
            print("Continuous listening mode enabled. Say 'stop' to end conversation.")
            stream = self.start_streaming_recording()

        try:
            while True:
                if use_mic:
                    if continuous:
                        # Continuous recording
                        time.sleep(2)  # Check every 2 seconds
                        audio_data = self.stop_streaming_recording(stream)
                        if len(audio_data) > 0:
                            user_text = self.transcribe_audio(audio_data)
                            if user_text:
                                print(f"\nYou said: {user_text}")
                        stream = self.start_streaming_recording()
                    else:
                        # Manual recording
                        print("\nPress Enter to start recording...")
                        input()
                        audio_data = self.record_audio(duration=5)
                        user_text = self.transcribe_audio(audio_data)
                        print(f"\nYou said: {user_text}")
                else:
                    # Text input mode
                    user_text = input("\nYou: ")

                if not user_text or not user_text.strip():
                    continue

                if user_text.lower() in ["exit", "quit", "bye", "stop"]:
                    print("\nGoodbye!")
                    break

                # Generate response
                print("\nThinking...")
                response = self.generate_response(user_text)
                print(f"Assistant: {response}")

                # Speak response
                print("Speaking...")
                self.speak(response)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            if (
                continuous
                and use_mic
                and hasattr(self, "is_recording")
                and self.is_recording
            ):
                self.stop_streaming_recording(stream)

            # Print performance stats
            if self.response_times:
                avg_time = sum(self.response_times) / len(self.response_times)
                print(f"\nAverage response time: {avg_time:.2f}s")
                print(f"Total responses: {len(self.response_times)}")

    def set_voice_settings(self, speaker=None, language=None, emotion=None, speed=6.0):
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
            try:
                # Get model manager
                tts_manager = TTS()
                # Get list of models
                models = tts_manager.list_models()

                print("\nAvailable Coqui TTS models:")

                # Check if it's a list or dictionary
                if isinstance(models, list):
                    for model in models:
                        print(f"  - {model}")
                elif isinstance(models, dict):
                    # New API returns dict with categories
                    for category, model_list in models.items():
                        print(f"\n{category}:")
                        for model in model_list:
                            print(f"  - {model}")
                else:
                    # Try to iterate anyway
                    print(f"  Models object type: {type(models)}")
                    # Try to get available models from the manager
                    available_models = tts_manager.get_available_models()
                    if available_models:
                        for model in available_models:
                            print(f"  - {model}")

                return models
            except Exception as e:
                print(f"Error listing Coqui models: {e}")
                return []
        return []


def __del__(self):
    """Cleanup"""
    try:
        if hasattr(self, "p") and self.tts_engine == "coqui":
            self.p.terminate()

        # Clear GPU memory safely
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass  # Ignore errors during cleanup


if __name__ == "__main__":
    # Choose your model (uncomment one):
    # MODEL_CHOICES:
    # 1. Small & Fast: "microsoft/phi-2" (2.7B)
    # 2. Balanced: "meta-llama/Llama-2-7b-chat-hf" (7B)
    # 3. Large: "openai/gpt-oss-20b" (20B) - requires 4-bit

    selected_model = "teknium/OpenHermes-2.5-Mistral-7B"  # Recommended for voice

    # In initialization:
    companion = VoiceCompanion(
        tts_engine="coqui",
        coqui_model="tts_models/en/thorsten/tacotron2-DDC",  # Faster, multi-speaker
        llm_model=selected_model,
        use_4bit=True,
    )

    # Set to female voice
    companion.set_voice_settings(speaker="p294", speed=1.5)

    # Optional: List available TTS models
    companion.list_coqui_models()

    # Optional: Change voice settings
    # companion.set_voice_settings(speed=1.1)

    # Run conversation
    # Parameters:
    # use_mic=True: Use microphone input
    # continuous=True: Continuous listening (experimental)
    companion.run_conversation(use_mic=True, continuous=False)
