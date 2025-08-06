# 🚀 Space Station Object Detection with YOLOv8 + AI Voice Assistant

This project leverages **Ultralytics YOLOv8** to detect critical tools and hazards inside a synthetic **space station environment**, combined with an **AI-powered voice assistant** using **Vosk speech recognition** and **Groq LLM** for real-time astronaut assistance. The system is optimized for **high accuracy** and **efficient inference** on edge devices like the **Raspberry Pi**.

---

## 📁 Project Overview

- **Framework**: YOLOv8 by Ultralytics + PyQt5 GUI
- **Dataset**: Falcon's Digital Twin (Space Station Environment)
- **AI Assistant**: Vosk Speech Recognition + Groq API (Llama3-70B)
- **Target**: Real-time object detection + voice-controlled medical assistance
- **Hardware Optimization**: Raspberry Pi (Edge deployment)

---

## ✨ Key Features

🎯 **Real-time Object Detection**: Detect tools, hazards, and critical objects in space station environments  
🗣️ **Voice Recognition**: Hands-free interaction using Vosk speech recognition  
🤖 **AI Medical Assistant**: Groq-powered LLM for emergency medical guidance  
📱 **Modern GUI**: PyQt5 interface with live camera feed and controls  
🔊 **Text-to-Speech**: Audio feedback for astronaut assistance  
🎥 **Multi-Model Support**: YOLOv8n, YOLOv8s, and pose estimation variants

---

## 🧪 1. Environment & Dependency Setup

### ✅ Prerequisites
- Python 3.8+ (via **Anaconda**)
- OS: Windows / MacOS / Linux
- Webcam or video input device
- Internet connection (for Groq API)

### 📦 Required Dependencies
```bash
# Core ML/CV libraries
ultralytics
opencv-python
torch
numpy

# GUI and Audio
PyQt5
pyttsx3
sounddevice

# Speech Recognition
vosk

# API requests
requests
python-dotenv
```

### 🚀 Setup Instructions

#### 1. Clone this Repository:
```bash
git clone https://github.com/Yajat-prabhakar/yolo-testing.git
cd yolo-testing
```

#### 2. Install Dependencies:
```bash
pip install ultralytics opencv-python torch numpy PyQt5 pyttsx3 sounddevice vosk requests python-dotenv
```

#### 3. Set up Environment Variables:
Create a `.env` file in the project root:
```bash
# .env file
GROQ_API_KEY=your-groq-api-key-here
```

> 🔑 **Get your Groq API key**: Visit [https://console.groq.com/](https://console.groq.com/) to obtain your free API key

#### 4. Download Required Models:
- **YOLOv8 Models**: Already included (yolov8n.pt, yolov8s.pt, etc.)
- **Vosk Speech Models**: See [Model/README.md](Model/README.md) for download instructions

---

## 🖼️ 2. Model Setup

### 📁 YOLOv8 Models (Included):
- `yolov8n.pt` - Nano model (fastest, 6.5MB)
- `yolov8s.pt` - Small model (higher accuracy, 22.6MB)  
- `yolov8n-pose.pt` - Pose estimation nano (6.8MB)
- `yolov8s-pose.pt` - Pose estimation small (23.5MB)

### 🎤 Vosk Speech Models (Download Required):
See [Model/README.md](Model/README.md) for detailed setup instructions:
- `vosk-model-en-us-0.22` - High accuracy English model (~70MB)
- `vosk-model-small-en-us-0.15` - Faster, smaller model (~40MB)

---

## 🏃‍♂️ 3. Running the Application

### 🚀 Launch the Main Application:
```bash
python Main.py
```

### 🎮 Application Features:

#### **Object Detection Tab**:
- Real-time webcam feed with YOLOv8 detection
- Switch between different YOLO models
- Confidence threshold adjustment
- Save detection results

#### **Voice Assistant Tab**:
- Voice-activated medical assistance
- Speech-to-text using Vosk
- AI-powered responses via Groq (Llama3-70B)
- Text-to-speech feedback

#### **Controls**:
- Start/Stop camera feed
- Toggle voice recognition
- Model selection dropdown
- Confidence slider
- Save image functionality

---

## 🔊 4. Voice Assistant Usage

### 🎤 Activation:
1. Click "Start Voice Recognition" 
2. Speak clearly into your microphone
3. Ask medical or emergency questions
4. Receive AI-powered audio responses

### 💬 Example Voice Commands:
- *"What should I do for a space suit breach?"*
- *"How do I treat a burn in zero gravity?"*
- *"Emergency medical procedures for unconscious crew member"*
- *"First aid for eye injury from debris"*

### 🤖 AI Medical Assistant:
- Powered by **Groq's Llama3-70B-8192** model
- Specialized in **space medicine** and **astronaut emergencies**
- Provides step-by-step medical guidance
- Audio feedback for hands-free operation

---

## ⚙️ 5. Configuration

### 🔧 Model Configuration:
Edit model paths in `Main.py`:
```python
# Available models
self.models = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt", 
    "YOLOv8n-pose": "yolov8n-pose.pt",
    "YOLOv8s-pose": "yolov8s-pose.pt"
}
```

### 🎯 Detection Settings:
```python
# Confidence threshold (adjustable via GUI)
confidence = 0.5  # Default: 50%

# Model inference settings
results = model(frame, conf=confidence)
```

### 🗣️ Voice Settings:
```python
# Vosk model path
model_path = "Model/vosk-model-en-us-0.22"

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech rate
```

---

## 📊 6. Expected Performance

### 🎯 Object Detection Metrics:
| Model | Size | Speed | Accuracy (mAP@0.5) |
|-------|------|-------|-------------------|
| YOLOv8n | 6.5MB | ~45 FPS | 0.823+ |
| YOLOv8s | 22.6MB | ~30 FPS | 0.850+ |

### 🚀 System Requirements:
- **Minimum**: 4GB RAM, integrated graphics
- **Recommended**: 8GB RAM, dedicated GPU
- **Edge Device**: Raspberry Pi 4 (4GB+)

### 📱 Real-time Performance:
- **Camera Feed**: 30 FPS (1080p)
- **Detection Latency**: <50ms (GPU) / <200ms (CPU)
- **Voice Response**: 2-3 seconds (including API call)

---

## 🗂️ 7. Project Structure

```
yolo-testing/
├── 📄 README.md                 # This file
├── 🐍 Main.py                   # Main application entry point
├── 🔒 .env                      # Environment variables (API keys)
├── 📋 .gitignore               # Git ignore rules
├── 🤖 yolov8n.pt              # YOLOv8 nano model
├── 🤖 yolov8s.pt              # YOLOv8 small model  
├── 🤖 yolov8n-pose.pt         # YOLOv8 pose nano model
├── 🤖 yolov8s-pose.pt         # YOLOv8 pose small model
└── 📁 Model/                   # Speech recognition models
    └── 📖 README.md            # Model download instructions
```

---

## 🔧 8. Troubleshooting

### ❌ Common Issues:

#### **"No module named 'cv2'"**:
```bash
pip install opencv-python
```

#### **"GROQ_API_KEY not found"**:
1. Create `.env` file in project root
2. Add: `GROQ_API_KEY=your-actual-api-key`
3. Restart the application

#### **"Vosk model not found"**:
1. Check [Model/README.md](Model/README.md)
2. Download required Vosk models
3. Extract to `Model/` directory

#### **Camera not detected**:
```python
# Try different camera indices in Main.py
self.cap = cv2.VideoCapture(0)  # Try 0, 1, 2...
```

#### **Poor detection accuracy**:
- Increase confidence threshold
- Switch to YOLOv8s model (higher accuracy)
- Ensure good lighting conditions

---

## 🚀 9. Deployment & Edge Computing

### 🥧 Raspberry Pi Deployment:
```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv

# Create virtual environment  
python3 -m venv yolo_env
source yolo_env/bin/activate

# Install optimized packages
pip install ultralytics[cpu] opencv-python-headless
```

### ⚡ Performance Optimization:
- Use **YOLOv8n** for maximum speed
- Reduce input resolution: `640x640` → `416x416`
- Enable **TensorRT** optimization (NVIDIA devices)
- Use **ONNX** export for CPU inference

---

## 🔬 10. Technical Details

### 🧠 AI Architecture:
- **Object Detection**: YOLOv8 (PyTorch)
- **Speech Recognition**: Vosk (Offline)
- **Language Model**: Groq Llama3-70B (Cloud API)
- **Text-to-Speech**: pyttsx3 (Local)

### 📡 API Integration:
```python
# Groq API configuration
url = "https://api.groq.com/openai/v1/chat/completions"
model = "llama3-70b-8192"
system_prompt = "You are a medical assistant in space..."
```

### 🎛️ GUI Framework:
- **Framework**: PyQt5
- **Layout**: Tabbed interface
- **Components**: Video display, controls, voice feedback
- **Threading**: Separate threads for camera, voice, AI processing

---

## 📈 11. Future Enhancements

🔮 **Planned Features**:
- [ ] Multi-language support (Spanish, Russian, Chinese)
- [ ] Offline LLM integration (Llama2/Mistral)
- [ ] Custom space object training pipeline
- [ ] WebRTC streaming for remote monitoring
- [ ] Integration with space station sensors
- [ ] Advanced pose estimation for crew health monitoring

---

## 🤝 12. Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

---

## 📄 13. License & Credits

### 📜 License:
This project is licensed under the **MIT License** - see LICENSE file for details.

### 🙏 Acknowledgments:
- **Ultralytics** - YOLOv8 framework
- **Groq** - Lightning-fast LLM inference  
- **Vosk** - Offline speech recognition
- **Falcon Digital Twin** - Space station dataset
- **PyQt5** - GUI framework

---

## 👨‍💻 14. Contact & Support

**🚀 Team Leader**: Raunaq Adlakha  
📧 **Email**: [raunaq.adalkha@gmail.com](mailto:raunaq.adalkha@gmail.com)  
🔗 **GitHub**: [yolo-testing](https://github.com/Yajat-prabhakar/yolo-testing)  
💬 **Issues**: Report bugs and request features via GitHub Issues

---

## ⚠️ 15. Important Notes

> 🔑 **API Key Required**: Add your Groq API key to `.env` file to enable AI voice assistant functionality  
> 🎤 **Microphone Access**: Grant microphone permissions for voice recognition  
> 📹 **Camera Access**: Ensure webcam is connected and accessible  
> 🌐 **Internet Required**: Voice assistant needs internet for Groq API calls  
> 💾 **Model Downloads**: Vosk models must be downloaded separately (see Model/README.md)

---

**🚀 This project combines cutting-edge deep learning with edge device deployment to ensure real-time space station monitoring and AI-powered astronaut assistance!**

---

*Last Updated: January 2025*
