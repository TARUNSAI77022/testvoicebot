from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import requests
from groq import Groq  # Import Groq client
from gtts import gTTS  # For Text-to-Speech

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/voice/")
async def voice_chat(audio_file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{audio_file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await audio_file.read())
        
        # Transcribe the audio file using Groq's Speech-to-Text API
        transcript = transcribe_audio(temp_file_path)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        # Send the transcribed text to Groq's Chat Completion API
        chat_response = get_groq_chat_response(transcript)
        
        # Generate a voice response using gTTS
        tts = gTTS(chat_response, lang="en")
        tts_file_path = "response.mp3"
        tts.save(tts_file_path)
        
        # Return the text response and a URL to download the audio file
        return {
            "text": chat_response,
            "audio_url": "/download-audio/"
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {"text": "", "audio_url": None, "error": "An error occurred while processing your request."}

def transcribe_audio(file_path: str):
    """
    Sends the audio file to Groq's Speech-to-Text API and returns the transcript.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="distil-whisper-large-v3-en",  # Replace with the desired Speech-to-Text model
                response_format="verbose_json",
            )
        return transcription.text
    except Exception as e:
        print(f"Error calling Groq Speech-to-Text API: {e}")
        raise ValueError("Failed to transcribe audio.")

def get_groq_chat_response(prompt: str):
    """
    Sends the prompt to Groq's Chat Completion API and returns the response.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama3-70b-8192",  # Replace with the desired Chat Completion model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq Chat Completion API: {e}")
        raise ValueError("Failed to generate response.")

@app.get("/download-audio/")
async def download_audio():
    # Serve the generated audio file as a downloadable response
    if not os.path.exists("response.mp3"):
        return {"error": "Audio file not found."}
    
    with open("response.mp3", "rb") as audio_file:
        content = audio_file.read()
    
    # Clean up the audio file after serving it
    os.remove("response.mp3")
    
    return Response(content=content, media_type="audio/mpeg")