from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
from griddehaze_inference import grid_dehaze_video, dehaze_image_cv2_frame

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower()

    in_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(in_path)

    out_filename = f"dehazed_{filename}"
    out_path = os.path.join(OUTPUT_FOLDER, out_filename)

    # Dehaze based on file type
    if file_ext in ALLOWED_IMAGE_EXTENSIONS:
        dehaze_image(in_path, out_path)
        return render_template('result.html', input_media=in_path, output_media=out_path, is_video=False)

    elif file_ext in ALLOWED_VIDEO_EXTENSIONS:
        grid_dehaze_video(in_path, out_path)
        return render_template('result.html', input_media=in_path, output_media=out_path, is_video=True)
    
    else:
        return "Unsupported file type"

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=False)
