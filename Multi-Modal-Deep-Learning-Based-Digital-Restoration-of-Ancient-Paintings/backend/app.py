import os
import time
from flask import Flask, render_template, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
from Context_Encoder.context_encoder_predict1 import restore_image
from utils.generate_edges import create_edge_map
from utils.post_processing import enhance_image

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.secret_key = "ancient_art_secret"

# Configuration
# Paths relative to where app.py is executed (which is now backend/)
# BUT static folder is mapped to ../frontend/static via Flask.
# We need filesystem paths for saving files.
UPLOAD_FOLDER = os.path.join('..', 'frontend', 'static', 'uploads')
RESULTS_FOLDER = os.path.join('..', 'frontend', 'static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Paths for results
            timestamp = int(time.time())
            base_name = os.path.splitext(filename)[0]
            
            # 1. Generate Edge Map (Multi-Modal Feature)
            edge_filename = f"{base_name}_edge_{timestamp}.jpg"
            edge_path = os.path.join(app.config['RESULTS_FOLDER'], edge_filename)
            create_edge_map(upload_path, edge_path)
            
            # 2. Restore Image (Deep Learning)
            restored_filename = f"{base_name}_restored_{timestamp}.jpg"
            restored_path = os.path.join(app.config['RESULTS_FOLDER'], restored_filename)
            try:
                restore_image(upload_path, restored_path)
            except Exception as e:
                flash(f"Error during restoration: {e}")
                return redirect(request.url)

            # 3. Post-Processing (Enhancement)
            enhanced_filename = f"{base_name}_enhanced_{timestamp}.jpg"
            target_enhanced_path = os.path.join(app.config['RESULTS_FOLDER'], enhanced_filename)
            final_enhanced_path = enhance_image(restored_path, target_enhanced_path)

            return render_template('index.html', 
                                   original_image=url_for('static', filename=f'uploads/{filename}'),
                                   edge_map=url_for('static', filename=f'results/{edge_filename}'),
                                   restored_image=url_for('static', filename=f'results/{restored_filename}'),
                                   enhanced_image=url_for('static', filename=f'results/{enhanced_filename}'))
                                   
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
