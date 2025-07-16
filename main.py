import os
import unicodedata
import telebot
from flask import Flask, request
from dotenv import load_dotenv
import requests
import tempfile
from threading import Thread

from TTS.api import TTS
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.config.shared_configs import BaseDatasetConfig
from torch.serialization import add_safe_globals

# 📦 Torch config (important for XTTSv2)
add_safe_globals([
    XttsArgs,
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig
])

# 🌱 Load env
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
ALLOWED_IDS = os.getenv("ALLOWED_USER_IDS")
if not API_TOKEN or not ALLOWED_IDS:
    raise ValueError("Check .env for API_TOKEN and ALLOWED_USER_IDS")

ALLOWED_USER_IDS = [int(uid.strip()) for uid in ALLOWED_IDS.split(",")]

# ⚙️ Setup
bot = telebot.TeleBot(API_TOKEN)
app = Flask(__name__)

# 🧠 Load XTTSv2
print("🔊 Loading XTTSv2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# 🗣️ Speaker WAV file (from GitHub)
WAV_URL = "https://raw.githubusercontent.com/narangchouhan/xtts-telegram-bot/main/samples/your_voice.wav"
response = requests.get(WAV_URL)
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
    tmp_wav.write(response.content)
    speaker_wav_path = tmp_wav.name

# 🔠 Normalize Hindi
def normalize_hindi(text):
    return unicodedata.normalize("NFC", text)

def is_allowed(user_id):
    return user_id in ALLOWED_USER_IDS

# ✅ Commands
@bot.message_handler(commands=["start", "help"])
def welcome(message):
    if not is_allowed(message.from_user.id):
        return bot.reply_to(message, "🚫 Access denied.")
    bot.reply_to(message, "👋 Hindi mein koi bhi message bhejo — main uski awaaz bana dunga!")

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    user_id = message.from_user.id
    if not is_allowed(user_id):
        return bot.reply_to(message, "🚫 Access denied.")
    text = normalize_hindi(message.text)
    output_path = f"output_{user_id}.wav"
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language="hi",
            file_path=output_path
        )
        with open(output_path, "rb") as audio:
            bot.send_voice(message.chat.id, audio)
        os.remove(output_path)
    except Exception as e:
        print(f"Error: {e}")
        bot.reply_to(message, "⚠️ Voice generate karne mein error aaya.")

# 🌐 Flask endpoint for Telegram webhook (optional)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        update = telebot.types.Update.de_json(request.stream.read().decode("utf-8"))
        bot.process_new_updates([update])
        return "OK", 200
    return "🤖 Bot is running!", 200

# 🚀 Flask + Bot run in parallel threads (required for Render)
def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

def run_bot():
    print("🤖 Bot polling started...")
    bot.infinity_polling()

if __name__ == "__main__":
    Thread(target=run_flask).start()
    Thread(target=run_bot).start()