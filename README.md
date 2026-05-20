# 🎵 Emotion-Based Music Recommender

A modern Streamlit app that detects your facial emotion via webcam and recommends matching music — all running locally in VS Code.

---

## 🚀 Quick Start (3 steps)

### Step 1 — Install Python 3.9+
Make sure you have Python 3.9 or newer. Check with:
```bash
python --version
```

### Step 2 — Install dependencies
Open a terminal in this folder and run:
```bash
pip install -r requirements.txt
```
> This installs Streamlit, OpenCV, FER (emotion detector with bundled AI model), and TensorFlow.  
> First install takes 2-5 minutes (TensorFlow is large).

### Step 3 — Run the app
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 🎭 How It Works

| Step | What happens |
|------|-------------|
| 1 | Browser requests webcam access (click **Allow**) |
| 2 | Click the shutter button to take a snapshot |
| 3 | FER (Facial Expression Recognition) analyses your face using a MobileNet CNN |
| 4 | Dominant emotion + confidence scores are shown |
| 5 | 10 curated songs matching your emotion appear as YouTube links |
| 6 | Click any song → opens YouTube |

---

## 🎯 Supported Emotions

| Emotion | Songs included |
|---------|---------------|
| 😊 Happy | Upbeat pop, dance |
| 😢 Sad | Emotional ballads, indie |
| 😡 Angry | Rock, metal |
| 😐 Neutral | Lo-fi, ambient, pop |
| 😨 Fear | Dark ambient, alternative |
| 🤢 Disgust | Grunge, alternative rock |
| 😲 Surprise | EDM, high-energy pop |

---

## ⚙️ Customisation

### Change number of songs
Use the **sidebar slider** in the app (5–20 songs).

### Add your own songs
Open `app.py` and find the `MUSIC_DB` dictionary. Each emotion has a list of `{"name", "artist", "url"}` dicts. Add your own entries there.

### Add Spotify support (optional)
1. Create a Spotify Developer account at https://developer.spotify.com/
2. Create an app and get your `CLIENT_ID` and `CLIENT_SECRET`
3. Uncomment `spotipy` in `requirements.txt` and run `pip install spotipy`
4. The app is pre-wired for Spotify — just add your credentials to a `.env` file:
   ```
   SPOTIFY_CLIENT_ID=your_id_here
   SPOTIFY_CLIENT_SECRET=your_secret_here
   ```

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: fer` | Run `pip install fer tensorflow` |
| Camera not working | Make sure your browser has camera permissions |
| No face detected | Ensure good lighting and face centred in frame |
| TensorFlow install fails | Try `pip install tensorflow-cpu` instead |
| Slow first run | FER loads a model on startup — this is one-time only |

---

## 📁 Project Structure

```
emotion_music_app/
├── app.py              ← Main Streamlit app (all-in-one)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

No external datasets, no manual model downloads — everything is bundled!

---

## 🧰 Tech Stack

- **Streamlit** — Web UI + built-in webcam widget
- **FER** — Facial Expression Recognition (MobileNet CNN, no download needed)
- **OpenCV** — Image processing
- **Python** — Backend logic

---

## ☁️ Deploying to Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io/
3. Connect your GitHub repo and set `app.py` as the main file
4. Click **Deploy** — done!

> Note: Streamlit Cloud may have limited webcam support. For production, consider adding `streamlit-webrtc` (see `app.py` comments).
