# multi-Modal Deep Learning Based Digital Restoration of Ancient Paintings

**Final Year Engineering Project (Web Application)**

## 📖 Project Overview
This project is a **Full-Stack Web Application** designed for the digital preservation of cultural heritage. It uses a **Deep Learning-based Context Encoder (GAN)** to restore physical damage in ancient paintings.

The system is "Multi-Modal" because it utilizes:
1.  **RGB Visual Data**: The raw pixel information of the painting.
2.  **Structural Edge Maps**: An auxiliary modality (generated via Canny Edge Detection) to understand the geometry of lost regions.

## 🏗️ System Architecture

### Frontend (User Interface)
*   **Technologies**: HTML5, CSS3, JavaScript.
*   **Features**:
    *   Clean, museum-themed UI.
    *   Drag-and-drop Image Upload.
    *   Real-time processing feedback.
    *   **Dashboard view**: Displays Original, Edge Map, and Restored result side-by-side.

### Backend (Server)
*   **Framework**: Python Flask.
*   **Role**:
    1.  Receives the uploaded image.
    2.  **Pre-processing**: Generates the structural edge map (Modality 2).
    3.  **Inference**: Passes the image to the TensorFlow/Keras Deep Learning model.
    4.  **Response**: Returns the paths of the processed images to the frontend.

### Deep Learning Model
*   **Architecture**: Context Encoder (CNN Encoder-Decoder + GAN).
*   **Optimization**: Fully Convolutional bottleneck for CPU efficiency.
*   **Weights**: Pre-trained model loaded from `weights/weights.weights.h5`.

## 📂 Project Structure
```
/project-root
 ├── app.py                     # Flask Server Entry Point
 ├── Context_Encoder/
 │    └── context_encoder_predict1.py  # Deep Learning Inference Logic
 ├── utils/
 │    └── generate_edges.py     # Edge Map Generation (Multi-modal feature)
 ├── weights/
 │    └── weights.weights.h5    # Trained Model Weights
 ├── static/
 │    ├── uploads/              # Storage for user uploads
 │    └── results/              # Storage for generated results
 ├── templates/
 │    └── index.html            # Web Interface
 ├── README.md                  # Documentation
 └── requirements.txt           # Dependencies
```

## 🚀 How to Run the Website

### 1. Install Dependencies
Ensure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Run the Flask application:
```bash
python app.py
```

### 3. Access the App
Open your web browser and go to:
`http://127.0.0.1:5000`

### 4. Demo
1.  Upload a damaged painting image.
2.  Click **"Restore Painting"**.
3.  Wait for the AI to process (1-5 seconds).
4.  View the **Structural Edge Map** and the **Final Restoration**.

---
*Developed for Final Year Engineering Project.*
