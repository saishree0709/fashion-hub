import os
import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ------------------ Setup ------------------
load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ------------------ Skin Tone Detection ------------------
def detect_skin_tone(image_path):
    """
    Detects skin tone using YCrCb color space and brightness.
    Returns extended skin tone categories.
    """
    img = cv2.imread(image_path)
    if img is None:
        return "unknown"

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)

    skin_pixels = ycrcb[:, :, 0][skin_mask > 0]

    if len(skin_pixels) == 0:
        return "medium"

    avg_brightness = np.mean(skin_pixels)

    if avg_brightness > 200:
        return "very_light"
    elif avg_brightness > 170:
        return "light"
    elif avg_brightness > 130:
        return "medium"
    elif avg_brightness > 100:
        return "tan"
    elif avg_brightness > 70:
        return "deep"
    else:
        return "very_deep"


# ------------------ Color Palette ------------------
def get_color_palette(skin_tone):
    palettes = {
        "very_light": [
            {"name": "Soft Pink", "hex": "#F8C8DC"},
            {"name": "Sky Blue", "hex": "#87CEEB"},
            {"name": "Lavender", "hex": "#E6E6FA"},
            {"name": "Mint", "hex": "#98FF98"},
            {"name": "Beige", "hex": "#F5F5DC"},
        ],
        "light": [
            {"name": "Peach", "hex": "#FFE5B4"},
            {"name": "Coral", "hex": "#FF7F50"},
            {"name": "Light Grey", "hex": "#D3D3D3"},
            {"name": "Soft Yellow", "hex": "#FFFACD"},
            {"name": "Pastel Blue", "hex": "#AEC6CF"},
        ],
        "medium": [
            {"name": "Olive", "hex": "#708238"},
            {"name": "Mustard", "hex": "#FFDB58"},
            {"name": "Teal", "hex": "#008080"},
            {"name": "Warm Red", "hex": "#C04000"},
            {"name": "Cream", "hex": "#FFFDD0"},
        ],
        "tan": [
            {"name": "Burnt Orange", "hex": "#CC5500"},
            {"name": "Forest Green", "hex": "#228B22"},
            {"name": "Turquoise", "hex": "#40E0D0"},
            {"name": "Maroon", "hex": "#800000"},
            {"name": "Camel", "hex": "#C19A6B"},
        ],
        "deep": [
            {"name": "Royal Blue", "hex": "#4169E1"},
            {"name": "Emerald", "hex": "#50C878"},
            {"name": "Burgundy", "hex": "#800020"},
            {"name": "Gold", "hex": "#FFD700"},
            {"name": "Plum", "hex": "#8E4585"},
        ],
        "very_deep": [
            {"name": "Fuchsia", "hex": "#FF00FF"},
            {"name": "Electric Blue", "hex": "#7DF9FF"},
            {"name": "Bright White", "hex": "#FFFFFF"},
            {"name": "Crimson", "hex": "#DC143C"},
            {"name": "Orange", "hex": "#FFA500"},
        ],
    }

    return palettes.get(skin_tone, [])


# ------------------ Groq AI Styling ------------------
# ------------------ Groq AI Styling ------------------
def get_styling_recommendation(skin_tone, event_type=None):
    if not GROQ_API_KEY:
        return "Groq API Key missing. Please check .env file."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    if event_type:
        prompt = f"""
        Act as a professional fashion stylist.
        Event: {event_type}
        User's Skin Tone: {skin_tone}
        
        Provide 3 stylish, specific outfit recommendations perfect for this event and skin tone.
        For each, briefly mention colors and fabrics.
        Keep the tone fun and encouraging.
        """
    else:
        prompt = f"""
        Give 3 short, stylish fashion recommendations for someone with {skin_tone} skin tone.
        Include:
        - Outfit suggestion
        - Fabric suggestion
        - Color styling tip
        Keep it concise and modern.
        """

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200
    }

    try:
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Groq API Error: {response.text}"

    except Exception as e:
        return f"API Error: {str(e)}"


# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/styling')
def styling():
    return render_template('styling.html')

@app.route('/stylemap')
def stylemap():
    return render_template("stylemap.html")


@app.route('/gatherings')
def gatherings():
    return render_template("gatherings.html")


@app.route('/connect')
def connect():
    return render_template("connect.html")



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    skin_tone = detect_skin_tone(filepath)
    
    # Check for event type from form data (for gatherings page)
    event_type = request.form.get('event_type')
    
    recommendation = get_styling_recommendation(skin_tone, event_type)
    palette = get_color_palette(skin_tone)

    return jsonify({
        "skin_tone": skin_tone,
        "recommendation": recommendation,
        "color_palette": palette,
        "image_url": filepath
    })


# ------------------ Run App ------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)