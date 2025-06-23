# 🚀 GPU QuickCap - AI Video Captioning

A modern web application that automatically generates captions for videos using AI (OpenAI Whisper) with GPU acceleration. Perfect for creating social media content with 9:16 vertical format and beautiful overlay captions.

## ✨ Features

- **🤖 AI-Powered**: Uses OpenAI Whisper for accurate speech recognition
- **⚡ GPU Accelerated**: CUDA acceleration for both AI processing and video encoding
- **📱 Mobile Ready**: Automatically converts to 9:16 vertical format
- **🎨 Modern UI**: Beautiful, responsive web interface with drag & drop
- **🔄 Real-time Progress**: Live progress updates during processing
- **📥 Easy Download**: Direct download of processed videos

## 🛠️ Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (recommended for best performance)
- **FFmpeg with CUDA support** installed and in PATH
- **Arial font** (or modify `app.py` to use a different font)

## 📦 Installation

1. **Clone or download this repository**
```bash
git clone <your-repo-url>
cd "GPU quickcap"
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure FFmpeg with CUDA support is installed**
   - Download from: https://ffmpeg.org/download.html
   - Make sure it's in your system PATH
   - Verify CUDA support: `ffmpeg -hwaccels`

## 🚀 Quick Start

1. **Run the application**
```bash
python run.py
```

2. **Open your browser**
   - Web interface: http://localhost:8000
   - API documentation: http://localhost:8000/docs

3. **Upload a video**
   - Drag & drop or click to select a video file
   - Click "Generate Captions with AI"
   - Wait for processing to complete
   - Download your captioned video!

## 🎯 How It Works

1. **Upload**: User uploads a video file through the web interface
2. **Transcription**: OpenAI Whisper "small" model analyzes the audio and generates word-level timestamped transcriptions
3. **Caption Generation**: Text is chunked into 6-word phrases and rendered as wrapped PNG overlays (72pt Arial font)
4. **Video Processing**: FFmpeg combines the original video with caption overlays using CUDA acceleration (overlay_cuda)
5. **Format Conversion**: Video is cropped and scaled to 9:16 format (1080x1920) using scale_cuda
6. **Download**: User downloads the final processed video encoded with h264_nvenc

## ⚙️ Configuration

### Video Settings
- **Output format**: 9:16 vertical (1080x1920)
- **Codec**: H.264 with NVENC (GPU encoding)
- **Caption style**: Configurable templates including word-by-word highlighting
- **Current template**: Komikax font (65px) with cycling color word highlighting and 3px black stroke
- **Caption timing**: 6-word phrases with automatic word-level timing
- **Caption position**: Positioned at 70% from the top of the screen

### AI Model
- **Default model**: Whisper "small" (faster processing)
- **Alternative models**: Can be changed in `app.py` (tiny, base, small, medium, large)

### GPU Acceleration
- **Video encoding**: Uses `h264_nvenc` for GPU-accelerated encoding
- **Video processing**: Uses `scale_cuda` and `overlay_cuda` filters
- **AI processing**: Whisper automatically uses GPU if available

## 🔧 Customization

### Change Caption Template
Switch between different caption styles in `app.py`:
```python
CURRENT_TEMPLATE = "default"           # Basic white text
CURRENT_TEMPLATE = "komikax_highlight"  # Komikax with yellow highlighting
```

### Create Custom Template
Add your own template to `CAPTION_TEMPLATES`:
```python
"my_template": {
    "font_paths": ["fonts/MyFont.ttf"],
    "font_size": 80,
    "text_color": (255, 255, 255, 255),    # White
    "highlight_colors": [                  # Cycling colors for words
        (255, 255, 0, 255),   # Yellow
        (0, 255, 0, 255),     # Green  
        (255, 0, 0, 255)      # Red
    ],
    "line_spacing": 25,
    "stroke_color": (0, 0, 0, 255),        # Black stroke
    "stroke_width": 3                      # 3px stroke width
}
```

### Single Color vs Cycling Colors
```python
# Single color highlighting (old method)
"highlight_color": (255, 255, 0, 255)     # All words use yellow

# Cycling color highlighting (new method)  
"highlight_colors": [
    (255, 255, 0, 255),   # Word 1: Yellow
    (0, 255, 0, 255),     # Word 2: Green
    (255, 0, 0, 255)      # Word 3: Red, then repeats
]
```

### Change Caption Style
Modify existing template settings:
```python
LINE_SPACING = 30  # Space between lines in pixels
FONT_SIZE = 65     # Font size in points
```

### Change Caption Timing
Modify the `WORDS_PER_PHRASE` constant or `chunk_words()` function to change phrase length:
```python
WORDS_PER_PHRASE = 6  # Change to 2, 3, 4, 5, etc.
# or in chunk_words function:
if len(phrase) == WORDS_PER_PHRASE:
```

### Change Caption Position
Modify the `CAPTION_Y_POSITION` constant in `app.py`:
```python
CAPTION_Y_POSITION = 0.7  # 70% from top
CAPTION_Y_POSITION = 0.5  # Center of screen
CAPTION_Y_POSITION = 0.9  # Near bottom
```

### Change Video Output Format
Modify the crop and scale parameters in `app.py`:
```python
"crop=(in_h*9/16):in_h,scale_cuda=1080:1920"  # 9:16 format
"crop=in_w:in_h,scale_cuda=1920:1080"         # 16:9 format
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not found**
   - Install NVIDIA drivers and CUDA toolkit
   - Verify with: `nvidia-smi`

2. **FFmpeg errors**
   - Ensure FFmpeg is installed with CUDA support
   - Check PATH: `ffmpeg -version`

3. **Font not found**
   - Install Arial font or change font path in `app.py`
   - Use system fonts: `"C:/Windows/Fonts/arial.ttf"` (Windows)

4. **Memory issues**
   - Use smaller Whisper model: `whisper.load_model("tiny")`
   - Reduce video resolution in FFmpeg settings

### Performance Tips

- **Use SSD storage** for faster file I/O
- **Close other GPU applications** during processing
- **Use smaller Whisper models** for faster transcription
- **Optimize video file sizes** before upload

## 📝 API Documentation

The application provides a REST API with the following endpoints:

- `GET /` - Web interface
- `POST /upload/` - Upload and process video
- `GET /docs` - Interactive API documentation

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve GPU QuickCap!

## 📄 License

This project is open source. Please check the license file for details.

---

**Made with ❤️ and AI**