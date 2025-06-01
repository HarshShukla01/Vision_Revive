# 🌫️ VISION REVIVE: AI-Driven De-Smoking & De-Hazing Solution

> _A Final Year BTech CSE Project for Enhancing Visual Clarity in Emergency and Autonomous Systems_

---

![Banner](https://via.placeholder.com/1200x300.png?text=VISION+REVIVE+-+AI+Dehazing+%26+Desmoking+Solution)

## 🔍 Overview

**Vision Revive** is a web-based application that aims to **restore visibility** in images and videos affected by smoke or haze. Originally conceptualized for **rescue operations in fire-prone indoor environments** and for **road safety in hazy weather for autonomous vehicles**, the project explores both traditional image processing and modern AI-driven methods to enhance visual clarity.

Although the final implementation encountered challenges, the journey involved **learning, prototyping, and experimenting** with state-of-the-art algorithms like **Dark Channel Prior (DCP)** and **AOD-Net**. The project has been evaluated and **successfully presented** as a part of our academic curriculum.

---

## 🎯 Key Objectives

- ✅ Develop a real-time **de-smoking and de-hazing solution** for images and videos.
- ✅ Utilize **AI/ML models** (e.g., AOD-Net) for intelligent visibility restoration.
- ✅ Build a simple, user-friendly **web interface** using Flask.
- ✅ Enable **image upload**, processing, and result visualization.
- ✅ Explore **video enhancement** (under testing & research phase).

---

## 🛠️ Tech Stack

| Component         | Technology Used               |
|------------------|-------------------------------|
| Web Framework     | Flask (Python)                |
| Image Processing  | OpenCV, NumPy                 |
| AI Model          | AOD-Net (planned, not stable) |
| Traditional Method| Dark Channel Prior (DCP)      |
| Visualization     | Matplotlib, HTML/CSS          |
| Deployment        | Localhost / GitHub Pages      |

---

## 📦 Features

- 📤 Upload an image via browser
- 🌁 Apply dehazing using **Dark Channel Prior**
- 📈 Experimented with **CLAHE** for contrast enhancement
- ⚙️ Planned integration of AOD-Net for AI-based enhancement
- 🎞️ (Beta) Testing with video streams *(not stable)*

---

## 🚧 Limitations & Learnings

- ❌ AOD-Net integration faced technical issues (model loading/inference failure)
- ❌ Real-time video processing was computationally intensive
- ✅ Gained experience with web development and image processing pipelines
- ✅ Understood model deployment challenges in constrained environments

---

## 📷 Sample Result

| Original Image | After Dehazing (DCP) |
|----------------|----------------------|
| ![original](./static/sample_input.jpg) | ![output](./static/sample_output.jpg) |

---

## 📁 Project Structure

VisionRevive/
├── app.py # Flask app
├── static/
│ ├── sample_input.jpg
│ └── sample_output.jpg
├── templates/
│ └── index.html
├── dehaze.py # DCP-based implementation
├── aodnet_model.py # Placeholder for AI model
└── README.md


---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/VisionRevive.git
   cd VisionRevive
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the app**
    ```bash 
    python app.py
4. **Open your browser at** ```http://localhost:5000```

🧠 Future Scope
✅ Fixing and deploying AOD-Net or other pretrained models (e.g., DehazeFormer)

📹 Smooth video dehazing with hardware acceleration

🔗 Cloud deployment using Streamlit or Flask + Docker

🧪 Further optimization for real-time applications

🙌 Acknowledgements
Our Professors & Project Guides

OpenCV, PyTorch, and related open-source communities

GitHub Copilot and ChatGPT for collaborative guidance

✨ Final Note
While Vision Revive was not 100% functional, it stood as a symbol of our learning, exploration, and perseverance. We are proud to have completed the journey and shared our vision for a safer, clearer, and smarter world.

“Not every working prototype is a success, but every effort that teaches something is.”

