"""
Dental Caries Detection Application
====================================
An advanced Streamlit application for detecting dental caries in X-ray images
using YOLOv8 deep learning model.

Authors: Lior & Shaked
Version: 2.0
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime
from typing import Tuple, List, Optional
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration constants."""
    
    # File constraints
    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    MAX_IMAGE_DIMENSION = 4000
    RESIZE_THRESHOLD = 1280
    
    # Model settings
    MODEL_PATH = "best.pt"
    DEFAULT_CONF_THRESHOLD = 0.25
    INTERNAL_CONF_THRESHOLD = 0.1
    
    # Colors (BGR format for OpenCV)
    COLOR_HIGH_CONF = (0, 255, 0)      # Green: 80-100%
    COLOR_MED_CONF = (0, 255, 255)     # Yellow: 60-80%
    COLOR_LOW_CONF = (0, 140, 255)     # Orange: 40-60%
    COLOR_VERY_LOW = (0, 0, 255)       # Red: <40%
    
    # UI Settings
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
    

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Dental Caries Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stDownloadButton button {
        width: 100%;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_box_color(confidence: float) -> Tuple[int, int, int]:
    """
    Get bounding box color based on confidence level.
    
    Args:
        confidence: Detection confidence score (0.0 to 1.0)
        
    Returns:
        BGR color tuple for OpenCV
    """
    if confidence >= 0.80:
        return Config.COLOR_HIGH_CONF
    elif confidence >= 0.60:
        return Config.COLOR_MED_CONF
    elif confidence >= 0.40:
        return Config.COLOR_LOW_CONF
    else:
        return Config.COLOR_VERY_LOW


def validate_image(uploaded_file) -> Image.Image:
    """
    Validate and load uploaded image file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If file is invalid, too large, or wrong format
    """
    # Check file size
    if uploaded_file.size > Config.MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"❌ File too large! Maximum size: {Config.MAX_FILE_SIZE_MB}MB. "
            f"Your file: {uploaded_file.size / 1024 / 1024:.1f}MB"
        )
    
    # Try to open image
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        raise ValueError(f"❌ Invalid image file: {str(e)}")
    
    # Check dimensions
    if max(image.size) > Config.MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"❌ Image too large! Maximum dimension: {Config.MAX_IMAGE_DIMENSION}px. "
            f"Your image: {max(image.size)}px"
        )
    
    return image


def resize_if_needed(image: Image.Image, max_size: int = Config.RESIZE_THRESHOLD) -> Tuple[Image.Image, bool]:
    """
    Resize image if it exceeds maximum size while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension in pixels
        
    Returns:
        Tuple of (resized image, was_resized boolean)
    """
    original_size = image.size
    if max(image.size) > max_size:
        image_copy = image.copy()
        image_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image_copy, True
    return image, False


def convert_to_bgr(img_array: np.ndarray) -> np.ndarray:
    """
    Convert image array to BGR format for OpenCV processing.
    
    Args:
        img_array: NumPy array of image
        
    Returns:
        BGR format NumPy array
    """
    if len(img_array.shape) == 2:  # Grayscale
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:  # RGBA
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def get_image_download_bytes(img_bgr: np.ndarray) -> bytes:
    """
    Convert BGR image to downloadable JPEG bytes.
    
    Args:
        img_bgr: BGR format NumPy array
        
    Returns:
        JPEG image as bytes
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=95)
    return buffered.getvalue()


def generate_json_report(
    image_name: str,
    detections_count: int,
    confidences: List[float],
    boxes: List[Tuple[int, int, int, int]],
    threshold: float
) -> str:
    """
    Generate detailed JSON report of detection results.
    
    Args:
        image_name: Name of the analyzed image
        detections_count: Number of detections found
        confidences: List of confidence scores
        boxes: List of bounding box coordinates
        threshold: Confidence threshold used
        
    Returns:
        JSON string of the report
    """
    report = {
        "metadata": {
            "image_name": image_name,
            "timestamp": datetime.now().isoformat(),
            "model": "YOLOv8 Dental Caries Detector",
            "confidence_threshold": threshold
        },
        "summary": {
            "total_detections": detections_count,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "max_confidence": float(np.max(confidences)) if confidences else 0.0,
            "min_confidence": float(np.min(confidences)) if confidences else 0.0
        },
        "detections": [
            {
                "id": i + 1,
                "confidence": float(conf),
                "confidence_percent": f"{conf * 100:.1f}%",
                "bounding_box": {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3])
                }
            }
            for i, (conf, box) in enumerate(zip(confidences, boxes))
        ]
    }
    return json.dumps(report, indent=2)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching."""
    try:
        model = YOLO(Config.MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info(f"Please ensure '{Config.MODEL_PATH}' is in the application directory.")
        st.stop()


# Load model
with st.spinner("🔄 Loading AI model..."):
    model = load_model()
    st.success("✅ Model loaded successfully!")


# ============================================================================
# USER INTERFACE
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🦷 Dental Caries Detection System</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-box">'
    '📸 Upload dental X-ray images to detect potential caries (cavities). '
    'The AI model will analyze the image and highlight areas of concern with color-coded confidence levels.'
    '</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Confidence threshold slider
    conf_threshold_percent = st.slider(
        "Confidence Threshold (%)",
        min_value=0,
        max_value=100,
        value=int(Config.DEFAULT_CONF_THRESHOLD * 100),
        help="Only show detections with confidence above this threshold"
    )
    conf_threshold = conf_threshold_percent / 100.0
    
    st.markdown("---")
    
    # Color legend
    st.subheader("🎨 Confidence Color Legend")
    st.markdown("""
    - 🟢 **Green**: 80-100% (High)
    - 🟡 **Yellow**: 60-80% (Medium)
    - 🟠 **Orange**: 40-60% (Low)
    - 🔴 **Red**: <40% (Very Low)
    """)
    
    st.markdown("---")
    
    # Info
    st.subheader("ℹ️ About")
    st.info(
        "This application uses YOLOv8 deep learning model "
        "trained on dental X-ray images to detect caries."
    )
    
    # Model info
    with st.expander("📊 Model Information"):
        st.write("**Model:** YOLOv8n")
        st.write("**mAP50:** 87.4%")
        st.write("**Precision:** 93.5%")
        st.write("**Recall:** 77.5%")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Choose a dental X-ray image...",
    type=Config.SUPPORTED_FORMATS,
    accept_multiple_files=False,
    help=f"Supported formats: {', '.join(Config.SUPPORTED_FORMATS).upper()}"
)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================


if uploaded_file is not None:
    try:
        # Progress tracking
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # Step 1: Validate image
        progress_bar.progress(20, text="Validating image...")
        image = validate_image(uploaded_file)
        
        # Step 2: Resize if needed
        progress_bar.progress(30, text="Optimizing image...")
        image, was_resized = resize_if_needed(image)
        if was_resized:
            st.info(f"ℹ️ Image resized to {image.size[0]}x{image.size[1]}px for optimal processing")
        
        # Step 3: Convert to BGR
        progress_bar.progress(40, text="Preparing image...")
        img_array = np.array(image)
        img_bgr = convert_to_bgr(img_array)
        
        # Step 4: Run detection
        progress_bar.progress(60, text="Running AI detection...")
        results = model(img_bgr, conf=Config.INTERNAL_CONF_THRESHOLD, verbose=False)
        
        # Step 5: Process results
        progress_bar.progress(80, text="Processing results...")
        annotated_img = img_bgr.copy()
        detections_count = 0
        confidences = []
        boxes = []
        
        # Draw boxes
        for box in results[0].boxes:
            score = box.conf.item()
            
            if score >= conf_threshold:
                detections_count += 1
                confidences.append(score)
                
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                
                # Get color
                color = get_box_color(score)
                
                # Draw rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"Caries {score:.0%}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(annotated_img, label_text, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        progress_bar.progress(100, text="Complete!")
        progress_bar.empty()
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        
        # Images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="📸 Original Image", use_container_width=True)
        
        with col2:
            result_image_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(
                result_image_rgb,
                caption=f"🔍 Detection Results (Threshold: {conf_threshold_percent}%)",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Results summary
        if detections_count == 0:
            st.success(f"✅ No caries detected above {conf_threshold_percent}% confidence threshold.")
        else:
            st.warning(f"⚠️ Found {detections_count} potential caries area(s)")
            
            # Statistics metrics
            st.subheader("📊 Detection Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Detections", detections_count)
            with col2:
                st.metric("Avg Confidence", f"{np.mean(confidences):.1%}")
            with col3:
                st.metric("Max Confidence", f"{np.max(confidences):.1%}")
            with col4:
                st.metric("Min Confidence", f"{np.min(confidences):.1%}")
            
            # Confidence distribution chart
            st.subheader("📈 Confidence Distribution")
            fig = px.histogram(
                confidences,
                nbins=10,
                labels={'value': 'Confidence Level', 'count': 'Number of Detections'},
                title="Distribution of Detection Confidence Scores"
            )
            fig.update_traces(marker_color='#1f77b4')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed detections table
            with st.expander("📋 Detailed Detection List"):
                detection_data = []
                for i, (conf, box) in enumerate(zip(confidences, boxes)):
                    detection_data.append({
                        "ID": i + 1,
                        "Confidence": f"{conf:.1%}",
                        "Location": f"({box[0]}, {box[1]}) → ({box[2]}, {box[3]})"
                    })
                st.table(detection_data)
        
        # ============================================================
        # DOWNLOAD OPTIONS
        # ============================================================
        
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download annotated image
            img_bytes = get_image_download_bytes(annotated_img)
            st.download_button(
                label="📥 Download Annotated Image",
                data=img_bytes,
                file_name=f"caries_detection_{uploaded_file.name}",
                mime="image/jpeg",
                use_container_width=True
            )
        
        with col2:
            # Download JSON report
            if detections_count > 0:
                json_report = generate_json_report(
                    uploaded_file.name,
                    detections_count,
                    confidences,
                    boxes,
                    conf_threshold
                )
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json_report,
                    file_name=f"report_{uploaded_file.name.rsplit('.', 1)[0]}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.button(
                    "📄 Download JSON Report",
                    disabled=True,
                    help="No detections to report",
                    use_container_width=True
                )
        
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.info("Please try with a different image or contact support.")


else:
    # Welcome message when no file is uploaded
    st.info("👆 Please upload one or more dental X-ray images to begin analysis")
    
    # Example usage
    with st.expander("💡 How to use this application"):
        st.markdown("""
        1. **Upload Image**: Click the upload button and select a dental X-ray image
        2. **Adjust Threshold**: Use the sidebar slider to filter detections by confidence level
        3. **View Results**: See the original and annotated images side-by-side
        4. **Review Statistics**: Check the detection metrics and confidence distribution
        5. **Download**: Save the annotated image and/or detailed JSON report
        
        **Tips:**
        - Supported formats: JPG, JPEG, PNG
        - Maximum file size: 10MB
        - For best results, use clear, high-quality X-ray images
        - Lower the confidence threshold to see more potential detections
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🦷 Dental Caries Detection System v2.0 | "
    "Powered by YOLOv8 | "
    "Made with ❤️ using Streamlit"
    "</div>",
    unsafe_allow_html=True
)
