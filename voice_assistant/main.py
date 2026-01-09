import queue
import threading

import numpy as np
import pyttsx3
import sounddevice as sd
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer


class VoiceCompanion:
    def __init__(self):
        # Load Whisper model
        self.whisper_model = whisper.load_model("base")

        # Load GPT-NeoX-20B (requires significant VRAM)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neox-20b", torch_dtype=torch.float16, device_map="auto"
        )

        # Initialize TTS
        self.tts_engine = pyttsx3.init()

        # Conversation history
        self.conversation_history = []

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
                **inputs, max_new_tokens=150, temperature=0.7, do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()

        # Update conversation history
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")

        return response

    def speak(self, text):
        """Convert text to speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def run_conversation(self):
        """Main conversation loop"""
        print("Voice Companion Ready! Press Ctrl+C to exit.")

        while True:
            # Record audio (simplified - use proper recording logic)
            print("Listening...")
            # Add audio recording logic here

            # Transcribe
            # user_text = self.transcribe_audio(audio_data)
            user_text = input("Type input (or implement audio recording): ")

            if user_text.lower() in ["exit", "quit"]:
                break

            # Generate response
            response = self.generate_response(user_text)
            print(f"Assistant: {response}")

            # Speak response
            self.speak(response)


if __name__ == "__main__":
    companion = VoiceCompanion()
    companion.run_conversation()
