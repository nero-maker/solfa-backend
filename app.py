from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from solfa_converter import convert_to_solfa  # Import your function

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return '🎵 Solfa Converter API is running! 🎶'

@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        solfa_lines = convert_to_solfa(filepath)
        return jsonify({'solfa': solfa_lines})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run( debug=False, host='0.0.0.0', port=port)
    
