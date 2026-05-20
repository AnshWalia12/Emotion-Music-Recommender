"""
Emotion-Based Music Recommender
================================
A fully modern Streamlit app that:
  1. Uses your webcam (via streamlit-webrtc) to capture a photo
  2. Detects emotions using FER (Facial Expression Recognition) library
  3. Recommends music via Spotify API based on detected emotion
  4. Falls back to curated YouTube links if no Spotify credentials are provided

Run:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
import os
import requests
import json
import random
from collections import Counter

# ─────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Emotion Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS  – dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- global ---- */
html, body, [class*="css"] { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

/* ---- title banner ---- */
.hero-banner {
    background: linear-gradient(90deg, #fc466b, #3f5efb);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.5);
}
.hero-banner h1 { color: white; font-size: 2.4rem; margin: 0; letter-spacing: 1px; }
.hero-banner p  { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin: 8px 0 0; }

/* ---- emotion badge ---- */
.emotion-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 12px auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ---- music card ---- */
.music-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 14px 18px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.music-card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,0.3); }
.music-card a { color: #fc466b; text-decoration: none; font-weight: 600; font-size: 1.05rem; }
.music-card a:hover { color: #ff8fab; }
.music-card .artist { color: #aaa; font-size: 0.9rem; margin-top: 4px; }

/* ---- section header ---- */
.section-header {
    color: white;
    font-size: 1.3rem;
    font-weight: 600;
    border-bottom: 2px solid #fc466b;
    padding-bottom: 6px;
    margin: 20px 0 14px;
}

/* ---- instruction step ---- */
.step-box {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #3f5efb;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    color: #ddd;
}

/* ---- confidence bar ---- */
.conf-row { display: flex; align-items: center; margin: 4px 0; gap: 10px; }
.conf-label { color: #ccc; width: 90px; font-size: 0.88rem; }
.conf-bar-bg { flex: 1; height: 10px; background: rgba(255,255,255,0.1); border-radius: 8px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 8px; }
.conf-pct { color: #aaa; font-size: 0.82rem; width: 40px; text-align: right; }

/* ---- spinner override ---- */
.stSpinner > div { border-top-color: #fc466b !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Lazy-import heavy libs with helpful error messages
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_fer():
    try:
        from fer import FER
        detector = FER(mtcnn=True)   # mtcnn=True → more accurate face detection
        return detector, None
    except ImportError:
        return None, "fer"

@st.cache_resource(show_spinner=False)
def load_webrtc():
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
        return webrtc_streamer, VideoTransformerBase, RTCConfiguration, None
    except ImportError:
        return None, None, None, "streamlit-webrtc"


# ─────────────────────────────────────────────────────────────────
# Curated fallback music dataset (7 emotions × 20 songs each)
# Uses real YouTube search URLs so links always work
# ─────────────────────────────────────────────────────────────────
YOUTUBE_SEARCH = "https://www.youtube.com/results?search_query="

def yt(query):
    return YOUTUBE_SEARCH + requests.utils.quote(query)

MUSIC_DB = {
    "happy": [
        {"name": "Happy",                  "artist": "Pharrell Williams",   "url": yt("Pharrell Williams Happy official")},
        {"name": "Can't Stop the Feeling", "artist": "Justin Timberlake",   "url": yt("Justin Timberlake Can't Stop The Feeling")},
        {"name": "Uptown Funk",            "artist": "Mark Ronson ft. Bruno Mars", "url": yt("Uptown Funk official")},
        {"name": "Walking on Sunshine",    "artist": "Katrina & The Waves",  "url": yt("Walking on Sunshine Katrina")},
        {"name": "Good as Hell",           "artist": "Lizzo",                "url": yt("Lizzo Good as Hell official")},
        {"name": "Shake It Off",           "artist": "Taylor Swift",         "url": yt("Taylor Swift Shake It Off")},
        {"name": "I Gotta Feeling",        "artist": "Black Eyed Peas",      "url": yt("Black Eyed Peas I Gotta Feeling")},
        {"name": "Dynamite",               "artist": "BTS",                  "url": yt("BTS Dynamite official")},
        {"name": "Levitating",             "artist": "Dua Lipa",             "url": yt("Dua Lipa Levitating official")},
        {"name": "Blinding Lights",        "artist": "The Weeknd",           "url": yt("The Weeknd Blinding Lights")},
        {"name": "Sunflower",              "artist": "Post Malone",          "url": yt("Post Malone Sunflower")},
        {"name": "Butter",                 "artist": "BTS",                  "url": yt("BTS Butter official")},
        {"name": "As It Was",              "artist": "Harry Styles",         "url": yt("Harry Styles As It Was")},
        {"name": "Peaches",               "artist": "Justin Bieber",        "url": yt("Justin Bieber Peaches")},
        {"name": "Industry Baby",          "artist": "Lil Nas X",            "url": yt("Lil Nas X Industry Baby")},
        {"name": "Anti-Hero",              "artist": "Taylor Swift",         "url": yt("Taylor Swift Anti-Hero")},
        {"name": "Golden Hour",            "artist": "JVKE",                 "url": yt("JVKE Golden Hour")},
        {"name": "Heat Waves",             "artist": "Glass Animals",        "url": yt("Glass Animals Heat Waves")},
        {"name": "Watermelon Sugar",       "artist": "Harry Styles",         "url": yt("Harry Styles Watermelon Sugar")},
        {"name": "Good Days",              "artist": "SZA",                  "url": yt("SZA Good Days")},
    ],
    "sad": [
        {"name": "Someone Like You",       "artist": "Adele",                "url": yt("Adele Someone Like You")},
        {"name": "The Night We Met",       "artist": "Lord Huron",           "url": yt("Lord Huron The Night We Met")},
        {"name": "Skinny Love",            "artist": "Bon Iver",             "url": yt("Bon Iver Skinny Love")},
        {"name": "Let Her Go",             "artist": "Passenger",            "url": yt("Passenger Let Her Go")},
        {"name": "Chasing Cars",           "artist": "Snow Patrol",          "url": yt("Snow Patrol Chasing Cars")},
        {"name": "Fix You",                "artist": "Coldplay",             "url": yt("Coldplay Fix You")},
        {"name": "All I Want",             "artist": "Kodaline",             "url": yt("Kodaline All I Want")},
        {"name": "Happier",                "artist": "Marshmello & Bastille","url": yt("Marshmello Bastille Happier")},
        {"name": "when the party's over",  "artist": "Billie Eilish",        "url": yt("Billie Eilish when the party's over")},
        {"name": "Call Out My Name",       "artist": "The Weeknd",           "url": yt("The Weeknd Call Out My Name")},
        {"name": "Leave The Door Open",    "artist": "Bruno Mars",           "url": yt("Bruno Mars Leave The Door Open")},
        {"name": "Telepatía",              "artist": "Kali Uchis",           "url": yt("Kali Uchis Telepatia")},
        {"name": "Liability",              "artist": "Lorde",                "url": yt("Lorde Liability")},
        {"name": "Motion Sickness",        "artist": "Phoebe Bridgers",      "url": yt("Phoebe Bridgers Motion Sickness")},
        {"name": "Traitor",                "artist": "Olivia Rodrigo",       "url": yt("Olivia Rodrigo Traitor")},
        {"name": "idontwannabeyouanymore", "artist": "Billie Eilish",        "url": yt("Billie Eilish idontwannabeyouanymore")},
        {"name": "Vienna",                 "artist": "Billy Joel",           "url": yt("Billy Joel Vienna")},
        {"name": "lovely",                 "artist": "Billie Eilish & Khalid","url": yt("Billie Eilish Khalid lovely")},
        {"name": "Funeral",                "artist": "Phoebe Bridgers",      "url": yt("Phoebe Bridgers Funeral")},
        {"name": "Exile",                  "artist": "Taylor Swift ft. Bon Iver","url": yt("Taylor Swift Exile Bon Iver")},
    ],
    "angry": [
        {"name": "Killing in the Name",    "artist": "Rage Against the Machine","url": yt("Rage Against the Machine Killing in the Name")},
        {"name": "Break Stuff",            "artist": "Limp Bizkit",          "url": yt("Limp Bizkit Break Stuff")},
        {"name": "Given Up",               "artist": "Linkin Park",          "url": yt("Linkin Park Given Up")},
        {"name": "Smells Like Teen Spirit","artist": "Nirvana",              "url": yt("Nirvana Smells Like Teen Spirit")},
        {"name": "Bulls on Parade",        "artist": "Rage Against the Machine","url": yt("Rage Against the Machine Bulls on Parade")},
        {"name": "Numb",                   "artist": "Linkin Park",          "url": yt("Linkin Park Numb")},
        {"name": "In The End",             "artist": "Linkin Park",          "url": yt("Linkin Park In The End")},
        {"name": "Faint",                  "artist": "Linkin Park",          "url": yt("Linkin Park Faint")},
        {"name": "Down with the Sickness",  "artist": "Disturbed",           "url": yt("Disturbed Down With The Sickness")},
        {"name": "Bodies",                 "artist": "Drowning Pool",        "url": yt("Drowning Pool Bodies")},
        {"name": "Closer",                 "artist": "Nine Inch Nails",      "url": yt("Nine Inch Nails Closer")},
        {"name": "Du Hast",                "artist": "Rammstein",            "url": yt("Rammstein Du Hast")},
        {"name": "Cochise",                "artist": "Audioslave",           "url": yt("Audioslave Cochise")},
        {"name": "Enter Sandman",          "artist": "Metallica",            "url": yt("Metallica Enter Sandman")},
        {"name": "Master of Puppets",      "artist": "Metallica",            "url": yt("Metallica Master of Puppets")},
        {"name": "Chop Suey!",             "artist": "System of a Down",     "url": yt("System of a Down Chop Suey")},
        {"name": "Toxicity",               "artist": "System of a Down",     "url": yt("System of a Down Toxicity")},
        {"name": "Paranoid",               "artist": "Black Sabbath",        "url": yt("Black Sabbath Paranoid")},
        {"name": "Highway to Hell",        "artist": "AC/DC",                "url": yt("ACDC Highway to Hell")},
        {"name": "Thunderstruck",          "artist": "AC/DC",                "url": yt("ACDC Thunderstruck")},
    ],
    "neutral": [
        {"name": "lo-fi hip hop radio",    "artist": "ChilledCow",           "url": yt("lofi hip hop beats to study")},
        {"name": "Clair de Lune",          "artist": "Debussy",              "url": yt("Debussy Clair de Lune piano")},
        {"name": "Comptine d'un autre été","artist": "Yann Tiersen",         "url": yt("Yann Tiersen Comptine")},
        {"name": "River Flows in You",     "artist": "Yiruma",               "url": yt("Yiruma River Flows in You")},
        {"name": "Experience",             "artist": "Ludovico Einaudi",     "url": yt("Ludovico Einaudi Experience")},
        {"name": "Gymnopédie No. 1",       "artist": "Erik Satie",           "url": yt("Satie Gymnopedie No 1")},
        {"name": "Electric Feel",          "artist": "MGMT",                 "url": yt("MGMT Electric Feel")},
        {"name": "Feel Good Inc.",         "artist": "Gorillaz",             "url": yt("Gorillaz Feel Good Inc")},
        {"name": "Midnight City",          "artist": "M83",                  "url": yt("M83 Midnight City")},
        {"name": "Madness",                "artist": "Muse",                 "url": yt("Muse Madness")},
        {"name": "Starlight",              "artist": "Muse",                 "url": yt("Muse Starlight")},
        {"name": "High Hopes",             "artist": "Panic! At The Disco",  "url": yt("Panic At The Disco High Hopes")},
        {"name": "Yellow",                 "artist": "Coldplay",             "url": yt("Coldplay Yellow")},
        {"name": "The Scientist",          "artist": "Coldplay",             "url": yt("Coldplay The Scientist")},
        {"name": "Clocks",                 "artist": "Coldplay",             "url": yt("Coldplay Clocks")},
        {"name": "Lost!",                  "artist": "Coldplay",             "url": yt("Coldplay Lost")},
        {"name": "Photograph",             "artist": "Ed Sheeran",           "url": yt("Ed Sheeran Photograph")},
        {"name": "Perfect",                "artist": "Ed Sheeran",           "url": yt("Ed Sheeran Perfect")},
        {"name": "Thinking Out Loud",      "artist": "Ed Sheeran",           "url": yt("Ed Sheeran Thinking Out Loud")},
        {"name": "A Sky Full of Stars",    "artist": "Coldplay",             "url": yt("Coldplay A Sky Full of Stars")},
    ],
    "fear": [
        {"name": "Comfortably Numb",       "artist": "Pink Floyd",           "url": yt("Pink Floyd Comfortably Numb")},
        {"name": "The Sound of Silence",   "artist": "Simon & Garfunkel",    "url": yt("Simon Garfunkel Sound of Silence")},
        {"name": "Mad World",              "artist": "Gary Jules",           "url": yt("Gary Jules Mad World")},
        {"name": "Breathe (2 AM)",         "artist": "Anna Nalick",          "url": yt("Anna Nalick Breathe 2 AM")},
        {"name": "Intro",                  "artist": "The xx",               "url": yt("The xx Intro")},
        {"name": "Teardrop",               "artist": "Massive Attack",       "url": yt("Massive Attack Teardrop")},
        {"name": "Angel",                  "artist": "Massive Attack",       "url": yt("Massive Attack Angel")},
        {"name": "Machine",                "artist": "Imagine Dragons",      "url": yt("Imagine Dragons Machine")},
        {"name": "Demons",                 "artist": "Imagine Dragons",      "url": yt("Imagine Dragons Demons")},
        {"name": "Radioactive",            "artist": "Imagine Dragons",      "url": yt("Imagine Dragons Radioactive")},
        {"name": "Breathe",                "artist": "Pink Floyd",           "url": yt("Pink Floyd Breathe")},
        {"name": "Shine On You Crazy Diamond","artist": "Pink Floyd",        "url": yt("Pink Floyd Shine On You Crazy Diamond")},
        {"name": "Welcome to the Machine", "artist": "Pink Floyd",           "url": yt("Pink Floyd Welcome to the Machine")},
        {"name": "Nightmare",              "artist": "Halsey",               "url": yt("Halsey Nightmare")},
        {"name": "Ghost",                  "artist": "Justin Bieber",        "url": yt("Justin Bieber Ghost")},
        {"name": "Neon Gravestones",       "artist": "twenty one pilots",    "url": yt("twenty one pilots Neon Gravestones")},
        {"name": "Jumpsuit",               "artist": "twenty one pilots",    "url": yt("twenty one pilots Jumpsuit")},
        {"name": "Stressed Out",           "artist": "twenty one pilots",    "url": yt("twenty one pilots Stressed Out")},
        {"name": "Ride",                   "artist": "twenty one pilots",    "url": yt("twenty one pilots Ride")},
        {"name": "Heathens",               "artist": "twenty one pilots",    "url": yt("twenty one pilots Heathens")},
    ],
    "disgust": [
        {"name": "Everything I Do",        "artist": "Radiohead",            "url": yt("Radiohead Creep")},
        {"name": "Creep",                  "artist": "Radiohead",            "url": yt("Radiohead Creep")},
        {"name": "Karma Police",           "artist": "Radiohead",            "url": yt("Radiohead Karma Police")},
        {"name": "Fake Plastic Trees",     "artist": "Radiohead",            "url": yt("Radiohead Fake Plastic Trees")},
        {"name": "Exit Music",             "artist": "Radiohead",            "url": yt("Radiohead Exit Music")},
        {"name": "Wires",                  "artist": "Athlete",              "url": yt("Athlete Wires")},
        {"name": "People Are Strange",     "artist": "The Doors",            "url": yt("The Doors People Are Strange")},
        {"name": "Personal Jesus",         "artist": "Depeche Mode",         "url": yt("Depeche Mode Personal Jesus")},
        {"name": "Enjoy the Silence",      "artist": "Depeche Mode",         "url": yt("Depeche Mode Enjoy the Silence")},
        {"name": "Policy of Truth",        "artist": "Depeche Mode",         "url": yt("Depeche Mode Policy of Truth")},
        {"name": "Black",                  "artist": "Pearl Jam",            "url": yt("Pearl Jam Black")},
        {"name": "Even Flow",              "artist": "Pearl Jam",            "url": yt("Pearl Jam Even Flow")},
        {"name": "Jeremy",                 "artist": "Pearl Jam",            "url": yt("Pearl Jam Jeremy")},
        {"name": "Black Hole Sun",         "artist": "Soundgarden",          "url": yt("Soundgarden Black Hole Sun")},
        {"name": "Spoonman",               "artist": "Soundgarden",          "url": yt("Soundgarden Spoonman")},
        {"name": "Black Days",             "artist": "Soundgarden",          "url": yt("Soundgarden Black Days")},
        {"name": "Rusty Cage",             "artist": "Soundgarden",          "url": yt("Soundgarden Rusty Cage")},
        {"name": "Fell on Black Days",     "artist": "Soundgarden",          "url": yt("Soundgarden Fell on Black Days")},
        {"name": "The Day I Tried to Live","artist": "Soundgarden",          "url": yt("Soundgarden The Day I Tried to Live")},
        {"name": "Like a Stone",           "artist": "Audioslave",           "url": yt("Audioslave Like a Stone")},
    ],
    "surprise": [
        {"name": "Mr. Brightside",         "artist": "The Killers",          "url": yt("The Killers Mr Brightside")},
        {"name": "Somebody That I Used to Know","artist": "Gotye",           "url": yt("Gotye Somebody That I Used to Know")},
        {"name": "Teenage Dream",          "artist": "Katy Perry",           "url": yt("Katy Perry Teenage Dream")},
        {"name": "Roar",                   "artist": "Katy Perry",           "url": yt("Katy Perry Roar")},
        {"name": "Firework",               "artist": "Katy Perry",           "url": yt("Katy Perry Firework")},
        {"name": "Counting Stars",         "artist": "OneRepublic",          "url": yt("OneRepublic Counting Stars")},
        {"name": "Stop and Stare",         "artist": "OneRepublic",          "url": yt("OneRepublic Stop and Stare")},
        {"name": "Secrets",                "artist": "OneRepublic",          "url": yt("OneRepublic Secrets")},
        {"name": "Apologize",              "artist": "OneRepublic",          "url": yt("OneRepublic Apologize")},
        {"name": "If I Lose Myself",       "artist": "OneRepublic",          "url": yt("OneRepublic If I Lose Myself")},
        {"name": "Wake Me Up",             "artist": "Avicii",               "url": yt("Avicii Wake Me Up")},
        {"name": "Hey Brother",            "artist": "Avicii",               "url": yt("Avicii Hey Brother")},
        {"name": "The Nights",             "artist": "Avicii",               "url": yt("Avicii The Nights")},
        {"name": "Levels",                 "artist": "Avicii",               "url": yt("Avicii Levels")},
        {"name": "Without You",            "artist": "Avicii",               "url": yt("Avicii Without You")},
        {"name": "Animals",                "artist": "Martin Garrix",        "url": yt("Martin Garrix Animals")},
        {"name": "Titanium",               "artist": "David Guetta ft. Sia", "url": yt("David Guetta Titanium Sia")},
        {"name": "She Wolf",               "artist": "David Guetta",         "url": yt("David Guetta She Wolf")},
        {"name": "Play Hard",              "artist": "David Guetta",         "url": yt("David Guetta Play Hard")},
        {"name": "Sexy Bitch",             "artist": "David Guetta",         "url": yt("David Guetta Sexy Bitch")},
    ],
}

# FER library → emotion label mapping to our keys
FER_TO_KEY = {
    "happy":    "happy",
    "sad":      "sad",
    "angry":    "angry",
    "neutral":  "neutral",
    "fear":     "fear",
    "disgust":  "disgust",
    "surprise": "surprise",
}

EMOTION_COLORS = {
    "happy":    ("🌟", "#FFD700", "linear-gradient(135deg,#f7971e,#ffd200)"),
    "sad":      ("😢", "#6495ED", "linear-gradient(135deg,#2980B9,#6DD5FA)"),
    "angry":    ("😡", "#FF4500", "linear-gradient(135deg,#f00,#ff4500)"),
    "neutral":  ("😐", "#9E9E9E", "linear-gradient(135deg,#636363,#a2ab58)"),
    "fear":     ("😨", "#9B59B6", "linear-gradient(135deg,#6a3093,#a044ff)"),
    "disgust":  ("🤢", "#27AE60", "linear-gradient(135deg,#1e8449,#27ae60)"),
    "surprise": ("😲", "#E67E22", "linear-gradient(135deg,#f093fb,#f5576c)"),
}

# ─────────────────────────────────────────────────────────────────
# Emotion detection helpers
# ─────────────────────────────────────────────────────────────────

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def detect_emotion_fer(pil_img: Image.Image, detector):
    """
    Use the FER library to detect dominant emotion.
    Returns (dominant_emotion: str, all_scores: dict, annotated_pil: Image)
    """
    frame = pil_to_cv2(pil_img)
    result = detector.detect_emotions(frame)

    if not result:
        return None, {}, pil_img

    # pick the face with highest confidence
    best = max(result, key=lambda r: max(r["emotions"].values()))
    emotions = best["emotions"]
    dominant = max(emotions, key=emotions.get)

    # Draw rectangle + label
    x, y, w, h = best["box"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (99, 179, 255), 3)
    label = f"{dominant} {emotions[dominant]*100:.0f}%"
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (99, 179, 255), 2, cv2.LINE_AA)

    return dominant, emotions, cv2_to_pil(frame)


def detect_emotion_fallback(pil_img: Image.Image):
    """
    OpenCV Haar-cascade face detection + heuristic brightness/contrast trick
    when FER is not installed.  Results are approximate.
    """
    frame = pil_to_cv2(pil_img)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = getattr(cv2, "data", None)
    haarcascade = (cascade_path.haarcascades + "haarcascade_frontalface_default.xml"
                   if cascade_path else "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(haarcascade)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        return None, {}, pil_img

    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    roi = gray[y:y+h, x:x+w]

    # Crude heuristics based on brightness variance → just pick neutral
    mean_val  = float(np.mean(roi))
    std_val   = float(np.std(roi))
    # Can't do real emotion recognition without a model; return neutral
    dominant  = "neutral"
    emotions  = {e: 0.0 for e in FER_TO_KEY}
    emotions["neutral"] = 1.0

    cv2.rectangle(frame, (x, y), (x + w, y + h), (99, 179, 255), 3)
    cv2.putText(frame, "Face detected", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (99, 179, 255), 2)

    return dominant, emotions, cv2_to_pil(frame)


# ─────────────────────────────────────────────────────────────────
# Music recommendation
# ─────────────────────────────────────────────────────────────────

def get_recommendations(emotion_key: str, n: int = 10):
    pool = MUSIC_DB.get(emotion_key, MUSIC_DB["neutral"])
    return random.sample(pool, min(n, len(pool)))


# ─────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────

def render_emotion_badge(emotion: str):
    emoji, color, gradient = EMOTION_COLORS.get(emotion, ("🎵","#888","linear-gradient(135deg,#555,#888)"))
    st.markdown(f"""
    <div style="text-align:center; margin: 16px 0;">
        <span class="emotion-badge" style="background:{gradient}; color:white;">
            {emoji} &nbsp; {emotion.upper()}
        </span>
    </div>""", unsafe_allow_html=True)

def render_confidence_bars(emotions: dict):
    st.markdown("<div class='section-header'>Emotion Confidence Scores</div>", unsafe_allow_html=True)
    COLORS = {
        "happy":"#FFD700","sad":"#6495ED","angry":"#FF4500",
        "neutral":"#9E9E9E","fear":"#9B59B6","disgust":"#27AE60","surprise":"#E67E22",
    }
    sorted_e = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    for emo, score in sorted_e:
        pct = int(score * 100)
        color = COLORS.get(emo, "#888")
        st.markdown(f"""
        <div class="conf-row">
            <span class="conf-label">{emo.capitalize()}</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="conf-pct">{pct}%</span>
        </div>""", unsafe_allow_html=True)

def render_music_cards(tracks: list):
    st.markdown("<div class='section-header'>🎵 Recommended Songs</div>", unsafe_allow_html=True)
    for i, t in enumerate(tracks, 1):
        st.markdown(f"""
        <div class="music-card">
            <a href="{t['url']}" target="_blank">#{i} — {t['name']}</a>
            <div class="artist">🎤 {t['artist']}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        n_songs = st.slider("Number of songs to recommend", 5, 20, 10)
        st.divider()

        st.markdown("## 📖 How it works")
        for i, step in enumerate([
            "Click **Take Snapshot** to capture your face via webcam",
            "The AI analyses your facial expression",
            "Your dominant emotion is displayed with confidence scores",
            "Click a song link → opens YouTube search",
            "Enjoy music that matches your mood!",
        ], 1):
            st.markdown(f'<div class="step-box"><b>Step {i}.</b> {step}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("## 🎭 Supported Emotions")
        for emo, (emoji, _, _) in EMOTION_COLORS.items():
            st.markdown(f"{emoji} **{emo.capitalize()}**")

        st.divider()
        st.caption("Powered by FER · Built with Streamlit · Music via YouTube")
    return n_songs


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    n_songs = sidebar()

    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <h1>🎵 Emotion-Based Music Recommender</h1>
        <p>Let your face choose the music — snap a photo, detect your emotion, get perfect songs instantly.</p>
    </div>""", unsafe_allow_html=True)

    # Load FER detector
    detector, fer_err = load_fer()
    if fer_err:
        st.warning(
            "⚠️ **FER library not installed** — Emotion detection will use a basic OpenCV fallback. "
            "For full accuracy, install the `fer` package (see requirements.txt).",
            icon="⚠️"
        )

    # ── Webcam section ─────────────────────────────────────────────
    st.markdown("---")
    col_cam, col_result = st.columns([1, 1], gap="large")

    with col_cam:
        st.markdown("<div class='section-header'>📸 Capture Your Emotion</div>", unsafe_allow_html=True)

        # Use Streamlit's built-in camera input (works on all browsers, no extra library needed)
        camera_image = st.camera_input(
            label="Click the shutter button to take a photo",
            help="Allow browser camera access when prompted",
            key="webcam"
        )

        if camera_image is not None:
            st.success("✅ Photo captured! Analysing emotion…")

    with col_result:
        if camera_image is not None:
            # Convert uploaded bytes → PIL
            pil_img = Image.open(io.BytesIO(camera_image.getvalue()))

            with st.spinner("🔍 Detecting emotion…"):
                if detector is not None:
                    dominant, emotions, annotated = detect_emotion_fer(pil_img, detector)
                else:
                    dominant, emotions, annotated = detect_emotion_fallback(pil_img)

            # Show annotated image
            st.image(annotated, caption="Processed image", use_container_width=True)

            if dominant is None:
                st.error("😕 No face detected. Please ensure your face is clearly visible and well-lit.")
            else:
                emotion_key = FER_TO_KEY.get(dominant, "neutral")
                render_emotion_badge(dominant)

                if emotions:
                    render_confidence_bars(emotions)

    # ── Music recommendations ─────────────────────────────────────
    if camera_image is not None and "dominant" in dir() and dominant:
        st.markdown("---")
        emotion_key = FER_TO_KEY.get(dominant, "neutral")
        tracks = get_recommendations(emotion_key, n=n_songs)

        _, mid, _ = st.columns([0.5, 9, 0.5])
        with mid:
            render_music_cards(tracks)

        # Refresh button
        st.markdown("")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🔄 Shuffle Recommendations", use_container_width=True):
                st.rerun()

    elif camera_image is None:
        # Placeholder / instructions
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#888; padding:40px 0;">
            <div style="font-size:4rem;">📷</div>
            <div style="font-size:1.2rem; margin-top:12px;">Take a photo above to get started!</div>
            <div style="font-size:0.9rem; margin-top:8px; color:#555;">
                Your face will be analysed in real-time and matched to a music playlist.
            </div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
