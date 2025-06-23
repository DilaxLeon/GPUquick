import os
import shutil
import uuid
import whisper
import subprocess
import logging
import time
import torch
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont

# Import configuration
try:
    from config import get_ffmpeg_binary, COMMON_FFMPEG_PATHS, VIDEO_SETTINGS, CAPTION_SETTINGS
except ImportError:
    # Fallback if config.py is not available
    def get_ffmpeg_binary():
        return os.getenv("FFMPEG_BINARY", "ffmpeg")
    COMMON_FFMPEG_PATHS = []
    VIDEO_SETTINGS = {"width": 1080, "height": 1920, "bitrate": "5M", "preset": "fast", "audio_codec": "aac"}
    CAPTION_SETTINGS = {"max_width_percent": 0.8, "y_position": 0.7, "words_per_phrase": 6, "line_spacing": 30}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_quickcap.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPU QuickCap", description="AI-powered video captioning with GPU acceleration")
templates = Jinja2Templates(directory="templates")

# Load Whisper model with logging
print("🚀 Starting GPU QuickCap Application...")
logger.info("Starting GPU QuickCap Application")
print("📥 Loading Whisper AI model...")
logger.info("Loading Whisper 'small' model")
start_time = time.time()
model = whisper.load_model("small")  # Automatically uses CUDA if available
load_time = time.time() - start_time
print(f"✅ Whisper model loaded successfully in {load_time:.2f} seconds")
logger.info(f"Whisper model loaded successfully in {load_time:.2f} seconds")

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  Running on CPU (GPU not available)")
    logger.warning("Running on CPU (GPU not available)")

UPLOAD_DIR = "uploads"
CAPTION_DIR = "captions"

# FFmpeg Configuration
FFMPEG_BINARY = get_ffmpeg_binary()  # Get from configuration
FFMPEG_PATHS = [
    FFMPEG_BINARY,  # User-specified or configured
    "ffmpeg.exe",   # Windows executable in current directory
    "ffmpeg",       # System PATH
    *COMMON_FFMPEG_PATHS  # Paths from configuration
]

def find_ffmpeg_binary():
    """Find the best available FFmpeg binary"""
    print("🔍 Searching for FFmpeg binary...")
    logger.info("Searching for FFmpeg binary")
    
    for ffmpeg_path in FFMPEG_PATHS:
        try:
            # Test if the binary exists and works
            result = subprocess.run([ffmpeg_path, "-version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Extract version info
                version_line = result.stdout.split('\n')[0]
                print(f"✅ Found FFmpeg: {ffmpeg_path}")
                print(f"📋 Version: {version_line}")
                logger.info(f"FFmpeg found: {ffmpeg_path} - {version_line}")
                return ffmpeg_path
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"FFmpeg not found at {ffmpeg_path}: {e}")
            continue
    
    print("❌ FFmpeg not found! Please install FFmpeg or set FFMPEG_BINARY environment variable")
    logger.error("FFmpeg binary not found")
    raise Exception("FFmpeg not found. Please install FFmpeg or specify the path using FFMPEG_BINARY environment variable")

# Find and validate FFmpeg
FFMPEG_CMD = find_ffmpeg_binary()

# Create directories with logging
print("📁 Setting up directories...")
logger.info("Creating upload and caption directories")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CAPTION_DIR, exist_ok=True)
print(f"✅ Directories ready: {UPLOAD_DIR}/, {CAPTION_DIR}/")
logger.info(f"Directories created: {UPLOAD_DIR}/, {CAPTION_DIR}/")

# Settings (from configuration)
VIDEO_WIDTH = VIDEO_SETTINGS["width"]
VIDEO_HEIGHT = VIDEO_SETTINGS["height"]
MAX_WIDTH = int(VIDEO_WIDTH * CAPTION_SETTINGS["max_width_percent"])
FONT_SIZE = 72
WORDS_PER_PHRASE = CAPTION_SETTINGS["words_per_phrase"]
CAPTION_Y_POSITION = CAPTION_SETTINGS["y_position"]
LINE_SPACING = CAPTION_SETTINGS["line_spacing"]

# Caption Templates
CAPTION_TEMPLATES = {
    "default": {
        "name": "Default",
        "description": "Simple white text with no effects",
        "font_paths": [
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ],
        "font_size": 72,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_color": None,
        "line_spacing": 30,
        "stroke_color": None,  # No stroke
        "stroke_width": 0
    },
    "MrBeast": {
        "name": "MrBeast Style",
        "description": "Komikax font with cycling colors (Yellow→Green→Red), 3px stroke, and shadow",
        "font_paths": [
            "fonts/Komikax.ttf",
            "Komikax.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Cycling colors for words
            (255, 255, 0, 255),   # Yellow
            (0, 255, 0, 255),     # Green
            (255, 0, 0, 255)      # Red
        ],
        "line_spacing": 30,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4)  # Shadow offset (x, y) in pixels
    },
    "Bold Green": {
        "name": "Bold Green",
        "description": "Uni Sans Heavy font with bright green word highlighting and shadow",
        "font_paths": [
            "fonts/Uni Sans Heavy.otf",
            "Uni Sans Heavy.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Single bright green color for words
            (0, 255, 0, 255)   # Bright Green
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4)  # Shadow offset (x, y) in pixels
    },
    "Bold Sunshine": {
        "name": "Bold Sunshine",
        "description": "Theboldfont with bright yellow word highlighting, 2px outline, and extra large spacing",
        "font_paths": [
            "fonts/Theboldfont.ttf",
            "Theboldfont.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Single bright yellow color for words
            (255, 255, 0, 255)   # Bright Yellow
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 2,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4)  # Shadow offset (x, y) in pixels
    },
    "Premium Orange": {
        "name": "Premium Orange",
        "description": "Poppins Bold Italic with vibrant orange highlighting, uppercase text, and dynamic spacing",
        "font_paths": [
            "fonts/Poppins-BoldItalic.ttf",
            "Poppins-BoldItalic.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Single vibrant orange color for words
            (235, 91, 0, 255)   # Vibrant Orange (#EB5B00)
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "uppercase": True  # Convert all text to uppercase
    },
    "Minimal White": {
        "name": "Minimal White",
        "description": "SpiegelSans with clean white highlighting, minimal styling, and professional spacing",
        "font_paths": [
            "fonts/SpiegelSans.otf",
            "SpiegelSans.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Pure white color for words
            (255, 255, 255, 255)   # Pure White
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4)  # Shadow offset (x, y) in pixels
    },
    "Orange Meme": {
        "name": "Orange Meme",
        "description": "LuckiestGuy with uniform orange color, bold cartoon styling, and uppercase text",
        "font_paths": [
            "fonts/LuckiestGuy.ttf",
            "LuckiestGuy.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 140, 0, 255),  # Orange
        "highlight_colors": [  # Same orange color for uniform appearance
            (255, 140, 0, 255)   # Orange (same as text_color)
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "uppercase": True  # Convert all text to uppercase
    },
    "Cinematic Quote": {
        "name": "Cinematic Quote",
        "description": "Proxima Nova Alt Condensed Black Italic with bright yellow highlighting and title case",
        "font_paths": [
            "fonts/Proxima Nova Alt Condensed Black Italic.otf",
            "Proxima Nova Alt Condensed Black Italic.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Bright yellow for keywords
            (255, 255, 0, 255)   # Bright Yellow
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "title_case": True  # Convert all text to title case
    },
    "Word by Word": {
        "name": "Word by Word",
        "description": "Poppins Black Italic with word-by-word display, enhanced font size, and uniform white color",
        "font_paths": [
            "fonts/Poppins-BlackItalic.ttf",
            "Poppins-BlackItalic.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 75,
        "text_color": (255, 255, 255, 255),  # Pure White
        "highlight_colors": [  # Same white color for uniform appearance
            (255, 255, 255, 255)   # Pure White (same as text_color)
        ],
        "line_spacing": 50,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "word_by_word": True,  # Enable word-by-word functionality
        "enhanced_font_size": 82  # 10% increase (75 * 1.1 = 82.5, rounded to 82)
    },
    "esports_caption": {
        "name": "Esports Caption",
        "description": "Exo2-Black with vibrant red-orange highlighting, gaming-style effects, and uppercase text",
        "font_paths": [
            "fonts/Exo2-Black.ttf",
            "Exo2-Black.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # Pure White
        "highlight_colors": [  # Vibrant red-orange for keywords
            (255, 69, 0, 255)   # Red-Orange (#FF4500)
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 2,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "uppercase": True,  # Convert all text to uppercase
        "scale_effect": True,  # Enable scale effect for highlighted words
        "scale_factor": 1.15  # Scale highlighted words by 15% (65px -> 75px)
    },
    "explainer_pro": {
        "name": "Explainer Pro",
        "description": "Helvetica Rounded with semi-transparent orange highlight bars behind important words",
        "font_paths": [
            "fonts/HelveticaRoundedLTStd-Bd.ttf",
            "HelveticaRoundedLTStd-Bd.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # Pure White
        "highlight_colors": [  # Semi-transparent dark orange for highlight bars
            (255, 140, 0, 230)   # Dark Orange (#FF8C00) with 230 opacity
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 2,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "highlight_bars": True,  # Enable highlight bars instead of text color change
        "bar_padding": 8  # Padding around text for highlight bars
    },
    "Reaction Pop": {
        "name": "Reaction Pop",
        "description": "Proxima Nova Alt Condensed Black with vibrant red highlighting and title case formatting",
        "font_paths": [
            "fonts/Proxima Nova Alt Condensed Black.otf",
            "Proxima Nova Alt Condensed Black.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 70,
        "text_color": (255, 255, 255, 255),  # Pure White
        "highlight_colors": [  # Pure red for keywords
            (255, 0, 0, 255)   # Pure Red (#FF0000)
        ],
        "line_spacing": 45,  # Between 40px and 50px for optimal spacing
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": None,  # No shadow for clean look
        "shadow_offset": None,
        "title_case": True,  # Convert all text to title case
        "scale_effect": True,  # Enable scale effect for highlighted words
        "scale_factor": 1.15  # Scale highlighted words by 15% (70px -> 80px)
    }
}

# Current template (can be changed)
CURRENT_TEMPLATE = "MrBeast"

def get_font(template_name=None, word_by_word_mode=False):
    """Get the best available font for the specified template"""
    if template_name is None:
        template_name = CURRENT_TEMPLATE
    
    template = CAPTION_TEMPLATES.get(template_name, CAPTION_TEMPLATES["default"])
    font_paths = template["font_paths"]
    
    # Use enhanced font size for word-by-word mode if available
    if word_by_word_mode and template.get("word_by_word", False):
        font_size = template.get("enhanced_font_size", template["font_size"])
    else:
        font_size = template["font_size"]
    
    logger.debug(f"Loading font for template: {template_name}, size: {font_size}")
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                print(f"✅ Using font: {font_path} (size: {font_size})")
                logger.info(f"Font loaded: {font_path} (size: {font_size})")
                return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError) as e:
            logger.debug(f"Failed to load font {font_path}: {e}")
            continue
    
    # Fallback to default font
    try:
        print(f"⚠️  Using default font (size: {font_size})")
        logger.warning(f"Using default font fallback (size: {font_size})")
        return ImageFont.load_default()
    except Exception as e:
        print("⚠️  Warning: Could not load any font, using basic font")
        logger.error(f"Could not load any font: {e}")
        return ImageFont.load_default()

# Helper: Split into phrases of 6 words
def chunk_words(words):
    phrases = []
    phrase = []
    for word in words:
        phrase.append(word)
        if len(phrase) == WORDS_PER_PHRASE:
            phrases.append(phrase)
            phrase = []
    if phrase:
        phrases.append(phrase)
    return phrases

# Helper function to get text size (compatible with newer Pillow versions)
def get_text_size(draw, text, font):
    """Get text width and height, compatible with both old and new Pillow versions"""
    try:
        # Try new method first (Pillow 10.0.0+)
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]  # width, height
    except AttributeError:
        # Fallback to old method for older Pillow versions
        return draw.textsize(text, font=font)

# Helper function to draw text with stroke and shadow
def draw_text_with_stroke(draw, position, text, font, text_color, stroke_color=None, stroke_width=0, shadow_color=None, shadow_offset=None):
    """Draw text with optional shadow and stroke/outline"""
    x, y = position
    
    # Draw shadow first (behind everything)
    if shadow_color and shadow_offset:
        shadow_x = x + shadow_offset[0]
        shadow_y = y + shadow_offset[1]
        
        # Draw shadow stroke if stroke is enabled
        if stroke_color and stroke_width > 0:
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((shadow_x + dx, shadow_y + dy), text, font=font, fill=shadow_color)
        
        # Draw shadow text
        draw.text((shadow_x, shadow_y), text, font=font, fill=shadow_color)
    
    # Draw stroke around main text
    if stroke_color and stroke_width > 0:
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx != 0 or dy != 0:  # Don't draw at the center position yet
                    draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    
    # Draw the main text on top
    draw.text((x, y), text, font=font, fill=text_color)

# Wrap caption text and render PNG with word highlighting
def render_caption_png_wrapped(text, output_path, highlight_word_index=None, template_name=None):
    if template_name is None:
        template_name = CURRENT_TEMPLATE
    
    template = CAPTION_TEMPLATES.get(template_name, CAPTION_TEMPLATES["default"])
    
    logger.debug(f"Rendering caption: '{text[:30]}{'...' if len(text) > 30 else ''}' with template: {template_name}")
    
    # Convert text case if template requires it
    if template.get("uppercase", False):
        text = text.upper()
    elif template.get("title_case", False):
        text = text.title()
    
    # Check if this is word-by-word mode (for single word rendering)
    is_word_by_word = template.get("word_by_word", False) and len(text.split()) == 1
    font = get_font(template_name, word_by_word_mode=is_word_by_word)
    text_color = template["text_color"]
    # Support both single highlight_color and multiple highlight_colors
    highlight_colors = template.get("highlight_colors", [template.get("highlight_color")] if template.get("highlight_color") else [])
    line_spacing = template["line_spacing"]
    stroke_color = template.get("stroke_color")
    stroke_width = template.get("stroke_width", 0)
    shadow_color = template.get("shadow_color")
    shadow_offset = template.get("shadow_offset")
    
    image = Image.new("RGBA", (VIDEO_WIDTH, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    words = text.split()
    lines = []
    line_words = []
    line = ""
    
    # Track which words are on which lines
    word_line_mapping = []
    current_line_index = 0
    
    for word_idx, word in enumerate(words):
        test_line = f"{line} {word}".strip()
        w, _ = get_text_size(draw, test_line, font)
        if w <= MAX_WIDTH:
            line = test_line
            line_words.append(word_idx)
        else:
            if line:
                lines.append(line)
                for w_idx in line_words:
                    word_line_mapping.append(current_line_index)
                current_line_index += 1
                line_words = [word_idx]
                line = word
            else:
                line = word
                line_words.append(word_idx)
    
    if line:
        lines.append(line)
        for w_idx in line_words:
            word_line_mapping.append(current_line_index)

    total_height = sum([get_text_size(draw, l, font)[1] + line_spacing for l in lines])
    y_start = (300 - total_height) // 2

    # Render each line
    for line_idx, line in enumerate(lines):
        line_words = line.split()
        w, h = get_text_size(draw, line, font)
        x_start = (VIDEO_WIDTH - w) // 2
        y = y_start + line_idx * (h + line_spacing)
        
        # If no highlighting or highlight not on this line, render normally
        if not highlight_colors or highlight_word_index is None:
            draw_text_with_stroke(draw, (x_start, y), line, font, text_color, stroke_color, stroke_width, shadow_color, shadow_offset)
        else:
            # Render word by word with highlighting
            current_x = x_start
            word_index_in_text = 0
            
            # Find the starting word index for this line
            for i in range(len(word_line_mapping)):
                if word_line_mapping[i] == line_idx:
                    word_index_in_text = i
                    break
            
            for word_idx_in_line, word in enumerate(line_words):
                current_word_index = word_index_in_text + word_idx_in_line
                
                if current_word_index == highlight_word_index:
                    # Check if this template uses highlight bars
                    if template.get("highlight_bars", False):
                        # Draw highlight bar behind the word
                        color_index = highlight_word_index % len(highlight_colors)
                        bar_color = highlight_colors[color_index]
                        bar_padding = template.get("bar_padding", 8)
                        
                        # Calculate word dimensions
                        word_width_no_space, word_height = get_text_size(draw, word, font)
                        
                        # Draw rounded rectangle behind the word
                        bar_x1 = current_x - bar_padding
                        bar_y1 = y - bar_padding
                        bar_x2 = current_x + word_width_no_space + bar_padding
                        bar_y2 = y + word_height + bar_padding
                        
                        # Create a temporary image for the rounded rectangle with transparency
                        bar_img = Image.new("RGBA", (bar_x2 - bar_x1, bar_y2 - bar_y1), (0, 0, 0, 0))
                        bar_draw = ImageDraw.Draw(bar_img)
                        
                        # Draw rounded rectangle
                        corner_radius = 8
                        bar_draw.rounded_rectangle(
                            [(0, 0), (bar_x2 - bar_x1, bar_y2 - bar_y1)],
                            radius=corner_radius,
                            fill=bar_color
                        )
                        
                        # Paste the bar onto the main image
                        image.paste(bar_img, (bar_x1, bar_y1), bar_img)
                        
                        # Draw text in regular color (white) on top of the bar
                        draw_text_with_stroke(draw, (current_x, y), word, font, text_color, stroke_color, stroke_width, shadow_color, shadow_offset)
                        word_width, _ = get_text_size(draw, word + " ", font)
                    else:
                        # Use cycling colors for highlighted words (traditional highlighting)
                        color_index = highlight_word_index % len(highlight_colors)
                        color = highlight_colors[color_index]
                        
                        # Check if scale effect is enabled for this template
                        if template.get("scale_effect", False):
                            scale_factor = template.get("scale_factor", 1.2)
                            scaled_font_size = int(template["font_size"] * scale_factor)
                            scaled_font = get_font(template_name, word_by_word_mode=False)
                            
                            # Create scaled font
                            try:
                                font_path = None
                                for path in template["font_paths"]:
                                    if os.path.exists(path):
                                        font_path = path
                                        break
                                
                                if font_path:
                                    scaled_font = ImageFont.truetype(font_path, scaled_font_size)
                                else:
                                    scaled_font = font  # Fallback to regular font
                            except:
                                scaled_font = font  # Fallback to regular font
                            
                            # Calculate vertical offset to center the scaled word
                            regular_height = get_text_size(draw, word, font)[1]
                            scaled_height = get_text_size(draw, word, scaled_font)[1]
                            y_offset = (scaled_height - regular_height) // 2
                            
                            draw_text_with_stroke(draw, (current_x, y - y_offset), word, scaled_font, color, stroke_color, stroke_width, shadow_color, shadow_offset)
                            word_width, _ = get_text_size(draw, word + " ", scaled_font)
                        else:
                            draw_text_with_stroke(draw, (current_x, y), word, font, color, stroke_color, stroke_width, shadow_color, shadow_offset)
                            word_width, _ = get_text_size(draw, word + " ", font)
                else:
                    color = text_color
                    draw_text_with_stroke(draw, (current_x, y), word, font, color, stroke_color, stroke_width, shadow_color, shadow_offset)
                    word_width, _ = get_text_size(draw, word + " ", font)
                
                current_x += word_width

    image.save(output_path)
    logger.debug(f"Caption image saved: {output_path}")

@app.get("/", response_class=PlainTextResponse)
def read_root():
    return "API is running"

@app.get("/templates")
async def get_templates():
    """Get available caption templates"""
    template_list = []
    for key, template in CAPTION_TEMPLATES.items():
        template_info = {
            "id": key,
            "name": template.get("name", key),
            "description": template.get("description", "No description available"),
            "font_size": template["font_size"],
            "has_highlighting": bool(template.get("highlight_colors") or template.get("highlight_color")),
            "has_stroke": bool(template.get("stroke_color")),
            "has_shadow": bool(template.get("shadow_color"))
        }
        template_list.append(template_info)
    
    return {
        "templates": template_list,
        "current_template": CURRENT_TEMPLATE
    }

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), template: str = Form("MrBeast")):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"🎬 NEW VIDEO PROCESSING REQUEST - {timestamp}")
    print(f"{'='*60}")
    logger.info(f"New video processing request started")
    
    print(f"📁 File: {file.filename} ({file.content_type})")
    print(f"🎨 Template: {template}")
    logger.info(f"Processing file: {file.filename}, template: {template}")
    
    video_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
    output_path = os.path.join(UPLOAD_DIR, f"{video_id}_out.mp4")
    
    print(f"🆔 Video ID: {video_id}")
    print(f"📥 Input path: {input_path}")
    print(f"📤 Output path: {output_path}")
    logger.info(f"Video ID: {video_id}, Input: {input_path}, Output: {output_path}")

    # Save uploaded file
    print(f"💾 Saving uploaded file...")
    logger.info("Saving uploaded file to disk")
    file_save_start = time.time()
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file_save_time = time.time() - file_save_start
    
    # Get file size
    file_size = os.path.getsize(input_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✅ File saved successfully ({file_size_mb:.2f} MB) in {file_save_time:.2f} seconds")
    logger.info(f"File saved: {file_size_mb:.2f} MB in {file_save_time:.2f} seconds")

    # AI Transcription
    print(f"\n🤖 Starting AI transcription with Whisper...")
    logger.info("Starting Whisper AI transcription")
    transcription_start = time.time()
    
    result = model.transcribe(input_path, word_timestamps=True)
    transcription_time = time.time() - transcription_start
    
    segments = result["segments"]
    words = [word for segment in segments for word in segment["words"]]
    phrases = chunk_words(words)
    
    total_words = len(words)
    total_phrases = len(phrases)
    detected_language = result.get("language", "unknown")
    
    print(f"✅ Transcription completed in {transcription_time:.2f} seconds")
    print(f"🗣️  Language detected: {detected_language}")
    print(f"📝 Total words: {total_words}")
    print(f"📄 Total phrases: {total_phrases} (6 words per phrase)")
    logger.info(f"Transcription completed: {transcription_time:.2f}s, Language: {detected_language}, Words: {total_words}, Phrases: {total_phrases}")

    overlay_cmds = []
    input_files = []
    input_file_count = 1  # Start from 1 because 0 is the input video
    
    # Template Processing
    print(f"\n🎨 Processing caption template...")
    logger.info(f"Processing template: {template}")
    
    # Validate template selection
    if template not in CAPTION_TEMPLATES:
        print(f"⚠️  Template '{template}' not found, using MrBeast as fallback")
        logger.warning(f"Template '{template}' not found, using MrBeast fallback")
        template = "MrBeast"  # Default fallback
    
    selected_template = CAPTION_TEMPLATES.get(template, CAPTION_TEMPLATES["MrBeast"])
    # Check for both old and new highlighting systems
    use_highlighting = (selected_template.get("highlight_color") is not None or 
                       (selected_template.get("highlight_colors") and len(selected_template.get("highlight_colors", [])) > 0))
    
    print(f"🎨 Using caption template: {template} ({selected_template.get('name', template)})")
    print(f"✨ Word highlighting: {'Enabled' if use_highlighting else 'Disabled'}")
    if use_highlighting and selected_template.get("highlight_colors"):
        colors = len(selected_template.get("highlight_colors", []))
        if template == "MrBeast":
            print(f"🌈 Cycling through {colors} highlight colors (Yellow→Green→Red)")
        elif template == "Bold Green":
            print(f"🟢 Using {colors} highlight color (Bright Green)")
        elif template == "Bold Sunshine":
            print(f"🟡 Using {colors} highlight color (Bright Yellow)")
        elif template == "Premium Orange":
            print(f"🟠 Using {colors} highlight color (Vibrant Orange)")
        elif template == "Minimal White":
            print(f"⚪ Using {colors} highlight color (Pure White)")
        elif template == "Orange Meme":
            print(f"🧡 Using uniform orange color (all text highlighted)")
        elif template == "Cinematic Quote":
            print(f"🟡 Using {colors} highlight color (Bright Yellow for keywords)")
        elif template == "Word by Word":
            print(f"⚪ Using uniform white color (word-by-word display)")
        elif template == "esports_caption":
            scale_factor = selected_template.get("scale_factor", 1.0)
            print(f"🔴 Using {colors} highlight color (Red-Orange for gaming)")
            if selected_template.get("scale_effect", False):
                print(f"📏 Scale effect: {scale_factor}x size for highlighted words")
        elif template == "explainer_pro":
            print(f"🟠 Using {colors} highlight color (Semi-transparent orange bars)")
            if selected_template.get("highlight_bars", False):
                bar_padding = selected_template.get("bar_padding", 8)
                print(f"📊 Highlight bars: Enabled with {bar_padding}px padding")
        elif template == "Reaction Pop":
            scale_factor = selected_template.get("scale_factor", 1.0)
            print(f"🔴 Using {colors} highlight color (Pure Red for reactions)")
            if selected_template.get("scale_effect", False):
                print(f"📏 Scale effect: {scale_factor}x size for highlighted words")
        else:
            print(f"🌈 Using {colors} highlight color(s)")
    print(f"📝 Font size: {selected_template['font_size']}px")
    if selected_template.get("word_by_word", False):
        enhanced_size = selected_template.get("enhanced_font_size", selected_template['font_size'])
        print(f"🔤 Word-by-word mode: Enhanced font size {enhanced_size}px (+10%)")
    if selected_template.get("uppercase", False):
        print(f"🔤 Text case: UPPERCASE")
    elif selected_template.get("title_case", False):
        print(f"🔤 Text case: Title Case")
    stroke_info = f"{selected_template.get('stroke_width', 0)}px black" if selected_template.get('stroke_color') else 'None'
    print(f"🖌️  Text stroke: {stroke_info}")
    shadow_info = f"{selected_template.get('shadow_offset', (0,0))[0]}px offset" if selected_template.get('shadow_color') else 'None'
    print(f"🌑 Text shadow: {shadow_info}")
    
    # Caption Generation
    print(f"\n🖼️  Generating caption images...")
    logger.info("Starting caption image generation")
    caption_generation_start = time.time()
    
    # Check if this is word-by-word template
    is_word_by_word_template = selected_template.get("word_by_word", False)
    if is_word_by_word_template:
        print(f"🔤 Word-by-word mode: Creating individual word captions")
        logger.info("Using word-by-word caption generation mode")
        # For word-by-word template, create individual word captions
        word_counter = 0
        for phrase_idx, phrase in enumerate(phrases):
            print(f"📝 Processing phrase {phrase_idx + 1}/{total_phrases}: {len(phrase)} words")
    
            for word_idx, word in enumerate(phrase):
                word_text = word['word'].strip()
                word_start = word['start']
                word_end = word['end']
                
                caption_path = os.path.join(CAPTION_DIR, f"{video_id}_word_{word_counter}.png")
                render_caption_png_wrapped(word_text, caption_path, template_name=template)
                
                input_files.extend(["-i", caption_path])
                
                if input_file_count == 1:
                    overlay_cmds.append(f"[0:v][{input_file_count}:v] overlay=enable='between(t,{word_start},{word_end})':x=(W-w)/2:y=H*{CAPTION_Y_POSITION} [v{input_file_count}];")
                else:
                    overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] overlay=enable='between(t,{word_start},{word_end})':x=(W-w)/2:y=H*{CAPTION_Y_POSITION} [v{input_file_count}];")
                
                input_file_count += 1
                word_counter += 1
    else:
        # Standard phrase-based processing
        print(f"📄 Standard phrase mode: Creating phrase-based captions")
        logger.info("Using standard phrase-based caption generation")
        for phrase_idx, phrase in enumerate(phrases):
            text = " ".join([w['word'] for w in phrase]).strip()
            phrase_start = phrase[0]['start']
            phrase_end = phrase[-1]['end']
            
            print(f"📝 Processing phrase {phrase_idx + 1}/{total_phrases}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            if use_highlighting:
                # Generate caption with word-by-word highlighting
                for word_idx, word in enumerate(phrase):
                    word_start = word['start']
                    word_end = word['end']
                    
                    caption_path = os.path.join(CAPTION_DIR, f"{video_id}_p{phrase_idx}_w{word_idx}.png")
                    render_caption_png_wrapped(text, caption_path, highlight_word_index=word_idx, template_name=template)
                    
                    input_files.extend(["-i", caption_path])
                    
                    if input_file_count == 1:
                        overlay_cmds.append(f"[0:v][{input_file_count}:v] overlay=enable='between(t,{word_start},{word_end})':x=(W-w)/2:y=H*{CAPTION_Y_POSITION} [v{input_file_count}];")
                    else:
                        overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] overlay=enable='between(t,{word_start},{word_end})':x=(W-w)/2:y=H*{CAPTION_Y_POSITION} [v{input_file_count}];")
                    
                    input_file_count += 1
            else:
                # Generate static caption for the whole phrase
                caption_path = os.path.join(CAPTION_DIR, f"{video_id}_{phrase_idx}.png")
                render_caption_png_wrapped(text, caption_path, template_name=template)
                
                input_files.extend(["-i", caption_path])
                
                if input_file_count == 1:
                    overlay_cmds.append(f"[0:v][{input_file_count}:v] overlay=enable='between(t,{phrase_start},{phrase_end})':x=(W-w)/2:y=H*{CAPTION_Y_POSITION} [v{input_file_count}];")
                else:
                    overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] overlay=enable='between(t,{phrase_start},{phrase_end})':x=(W-w)/2:y=H*{CAPTION_Y_POSITION} [v{input_file_count}];")
                
                input_file_count += 1

    caption_generation_time = time.time() - caption_generation_start
    total_captions = input_file_count - 1
    print(f"✅ Caption generation completed in {caption_generation_time:.2f} seconds")
    print(f"🖼️  Generated {total_captions} caption images")
    logger.info(f"Caption generation completed: {caption_generation_time:.2f}s, {total_captions} images")

    # Video Processing
    print(f"\n🎬 Starting video processing with FFmpeg...")
    logger.info("Starting FFmpeg video processing")
    video_processing_start = time.time()

    last_output = f"[v{input_file_count-1}]" if overlay_cmds else "[0:v]"

    # Construct the complete filter chain including crop/scale and overlays
    video_filters = "crop=(in_h*9/16):in_h,scale=1080:1920[scaled];"
    if overlay_cmds:
        # Replace [0:v] with [scaled] in the first overlay command
        overlay_cmds[0] = overlay_cmds[0].replace("[0:v]", "[scaled]")
        complete_filter = video_filters + "".join(overlay_cmds)
    else:
        complete_filter = video_filters
        last_output = "[scaled]"

    # Since we're using timeline expressions (enable='between(t,...)'), we need to use CPU-based filters
    # CUDA filters don't support timeline expressions
    if use_highlighting:
        print(f"🎨 Using CPU processing for word-by-word highlighting (CUDA doesn't support timeline expressions)")
        logger.info("Using CPU processing for word-by-word highlighting")
        ffmpeg_cmd = [
            FFMPEG_CMD,
            "-i", input_path,
            *input_files,
            "-filter_complex", complete_filter,
            "-map", last_output, 
            "-map", "0:a",  # Copy audio from original video
            "-c:v", "libx264",
            "-c:a", VIDEO_SETTINGS["audio_codec"],
            "-preset", VIDEO_SETTINGS["preset"], 
            "-b:v", VIDEO_SETTINGS["bitrate"],
            output_path
        ]
    else:
        # Try CUDA-accelerated version for simple captions without timeline
        print(f"🚀 Using GPU-accelerated processing...")
        logger.info("Using GPU-accelerated processing")
        ffmpeg_cmd = [
            FFMPEG_CMD, "-hwaccel", "cuda", 
            "-i", input_path,
            *input_files,
            "-filter_complex", complete_filter.replace("scale=", "scale_cuda=").replace("overlay", "overlay_cuda"),
            "-map", last_output, 
            "-map", "0:a",  # Copy audio from original video
            "-c:v", "h264_nvenc",
            "-c:a", VIDEO_SETTINGS["audio_codec"],
            "-preset", VIDEO_SETTINGS["preset"], 
            "-b:v", VIDEO_SETTINGS["bitrate"],
            output_path
        ]

    print(f"🎬 FFmpeg command: {' '.join(ffmpeg_cmd)}")
    logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
    
    try:
        print(f"⚙️  Executing FFmpeg...")
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        video_processing_time = time.time() - video_processing_start
        
        print("✅ Video processing completed successfully")
        logger.info(f"FFmpeg processing completed successfully in {video_processing_time:.2f} seconds")
        
        # Get output file size
        output_size = os.path.getsize(output_path)
        output_size_mb = output_size / (1024 * 1024)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"🎉 PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📊 PROCESSING SUMMARY:")
        print(f"   📁 Input file: {file_size_mb:.2f} MB")
        print(f"   📁 Output file: {output_size_mb:.2f} MB")
        print(f"   ⏱️  File save: {file_save_time:.2f}s")
        print(f"   🤖 Transcription: {transcription_time:.2f}s")
        print(f"   🖼️  Caption generation: {caption_generation_time:.2f}s")
        print(f"   🎬 Video processing: {video_processing_time:.2f}s")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   🎨 Template: {template}")
        print(f"   📝 Words: {total_words}")
        print(f"   📄 Phrases: {total_phrases}")
        print(f"   🖼️  Captions: {total_captions}")
        print(f"{'='*60}")
        
        logger.info(f"Processing completed successfully - Total time: {total_time:.2f}s, Template: {template}, Words: {total_words}, Output: {output_size_mb:.2f}MB")
        
    except subprocess.CalledProcessError as e:
        video_processing_time = time.time() - video_processing_start
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"❌ PROCESSING FAILED!")
        print(f"{'='*60}")
        print(f"❌ FFmpeg error (exit code: {e.returncode})")
        print(f"⏱️  Failed after: {total_time:.2f}s")
        print(f"Command: {' '.join(e.cmd)}")
        
        logger.error(f"FFmpeg processing failed after {video_processing_time:.2f}s - Exit code: {e.returncode}")
        logger.error(f"FFmpeg command: {' '.join(e.cmd)}")
        
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
            logger.error(f"FFmpeg STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
            logger.error(f"FFmpeg STDERR: {e.stderr}")
        
        print(f"{'='*60}")
        raise Exception(f"Video processing failed: {e.stderr if e.stderr else 'Unknown error'}")

    return FileResponse(output_path, media_type="video/mp4", filename="captioned_9_16_video.mp4")

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting GPU QuickCap Server on port 8080...")
    logger.info("Starting GPU QuickCap Server on port 8080")
    print("📝 Logs will be saved to: gpu_quickcap.log")
    print("🌐 Access the web interface at: http://localhost:8080")
    print("📊 Server logs and processing details will appear below...")
    print("="*60)
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)