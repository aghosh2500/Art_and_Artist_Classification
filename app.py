import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import io

# ============================================================================
# Configuration
# ============================================================================
ART_MODEL_PATH = 'DenseNet201_25_0.0001.h5'
ARTIST_MODEL_PATH = 'swin_transformer_artist_model_epoch50_lr1e-04_acc_65.12.pth'

art_labels = ["ai_art", "human_art"]

# Color Scheme
PRIMARY_COLOR = "#1f77b4"  # Professional blue
SECONDARY_COLOR = "#ff7f0e"  # Accent orange
BACKGROUND_COLOR = "#f8f9fa"  # Light gray
TEXT_COLOR = "#2c3e50"  # Dark gray
SUCCESS_COLOR = "#27ae60"  # Green
WARNING_COLOR = "#e74c3c"  # Red

# ============================================================================
# Custom Streamlit Configuration
# ============================================================================
st.set_page_config(
    page_title="Art & Artist Classification",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling
st.markdown(f"""
<style>
    :root {{
        --primary-color: {PRIMARY_COLOR};
        --secondary-color: {SECONDARY_COLOR};
        --background-color: {BACKGROUND_COLOR};
        --text-color: {TEXT_COLOR};
    }}
    
    * {{
        color: {TEXT_COLOR};
    }}
    
    .main {{
        background-color: {BACKGROUND_COLOR};
    }}
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 1.1rem;
        font-weight: 600;
    }}
    
    .header-container {{
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }}
    
    .header-container h1 {{
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }}
    
    .header-container p {{
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }}
    
    .about-section {{
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid {PRIMARY_COLOR};
        margin: 1.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    .about-section h2 {{
        color: {PRIMARY_COLOR};
        margin-top: 0;
    }}
    
    .feature-container {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .feature-box {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-top: 4px solid {PRIMARY_COLOR};
    }}
    
    .feature-box h3 {{
        color: {PRIMARY_COLOR};
        margin-top: 0;
    }}
    
    .classifier-container {{
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .result-success {{
        background-color: #d4edda;
        border: 2px solid {SUCCESS_COLOR};
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}
    
    .result-info {{
        background-color: #d1ecf1;
        border: 2px solid {PRIMARY_COLOR};
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Model Classes and Utility Functions
# ============================================================================
class CustomSwinModel(nn.Module):
    def __init__(self, base_model, num_ftrs, num_classes):
        super(CustomSwinModel, self).__init__()
        self.base_model = base_model
        self.base_model.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@st.cache_resource
def load_art_model():
    try:
        return tf.keras.models.load_model(ART_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load art model: {e}")
        return None

@st.cache_resource
def load_artist_model():
    try:
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        checkpoint = torch.load(ARTIST_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        class_names = checkpoint['classes']
        base_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
        artist_model = CustomSwinModel(base_model, num_ftrs=base_model.head.in_features, num_classes=len(class_names))
        artist_model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        artist_model = artist_model.to(device)
        artist_model.eval()
        return artist_model, class_names, device
    except Exception as e:
        st.error(f"Failed to load artist model: {e}")
        return None, None, None

def art_predict_from_pil(pil_img, model):
    img = pil_img.resize((224,224)).convert('RGB')
    img_array = keras_image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    probs = prediction[0]
    idx = int(np.argmax(probs))
    label = art_labels[idx]
    confidence = float(probs[idx])
    return label, confidence

def artist_predict_topk(pil_img, artist_model, class_names, device, k=2):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    img = pil_img.convert('RGB')
    image_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = artist_model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        topk = torch.topk(probabilities, k)
        indices = topk.indices.cpu().numpy()
        values = topk.values.cpu().numpy()
    results = [(class_names[int(i)], float(v)) for i, v in zip(indices, values)]
    return results

# ============================================================================
# Page Functions
# ============================================================================
def home_page():
    """Landing page with welcome and about section"""
    st.markdown(f"""
    <div class="header-container">
        <h1>🎨 Art & Artist Classification</h1>
        <p>Detect AI-generated art and identify artistic styles with advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome Section
    st.markdown(f"""
    <div class="about-section">
        <h2>Welcome</h2>
        <p>This tool uses state-of-the-art deep learning models to analyze artwork and provide intelligent classifications. 
        Whether you're interested in understanding if an artwork is AI-generated or curious about artistic styles, our 
        classifier provides fast and accurate results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # How It Works
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-box">
            <h3>📸 Step 1: Upload an Image</h3>
            <p>Start by uploading an image of artwork you'd like to analyze. The tool accepts PNG, JPG, and JPEG formats.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-box">
            <h3>🤖 Step 2: AI Detection</h3>
            <p>Our first model analyzes the image to determine if it's AI-generated or created by a human artist.</p>
        </div>
        """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div class="feature-box">
            <h3>🎯 Step 3: Artist Classification</h3>
            <p>If the artwork is identified as AI-generated, our second model predicts the most likely generative artist or style.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="feature-box">
            <h3>📊 Step 4: Results</h3>
            <p>View detailed confidence scores and predictions to understand what the models detected in your artwork.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # About Section
    st.markdown(f"""
    <div class="about-section">
        <h2>About This Project</h2>
        <p>I'm <strong>Advait Ghosh</strong>, a high schooler from the Bay Area. This project uses deep learning models to classify art and 
        artists, addressing a critical issue facing the creative community today. The artist community has experienced tremendous stress in recent 
        years, with ongoing lawsuits highlighting the challenges they face in protecting their work and intellectual property.</p>
        
        <p>Beyond the broader impact, this project holds personal significance for me—I've been doing art my whole life, which has given me a deep 
        appreciation for the struggles artists encounter. This personal connection, combined with the urgent need to develop tools that can help 
        protect and attribute artistic work, motivated me to tackle this problem with machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h3>Ready to analyze artwork?</h3>
        <p>Head over to the <b>Classifier</b> tab to get started!</p>
    </div>
    """, unsafe_allow_html=True)

def classifier_page():
    """Classifier page for image analysis"""
    st.markdown(f"""
    <div style="padding: 1.5rem 0; text-align: center;">
        <h1 style="color: {PRIMARY_COLOR}; margin: 0;">🎨 Classifier</h1>
        <p style="color: #666; margin: 0.5rem 0 0 0;">Upload an image to analyze its artistic classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"""
    <div class="classifier-container">
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil_img, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            st.write("**Image Details:**")
            st.write(f"- Size: {pil_img.size[0]} × {pil_img.size[1]} pixels")
            st.write(f"- Format: {pil_img.format or 'Unknown'}")
        
        st.markdown("---")
        
        # Load and run art model
        art_model = load_art_model()
        if art_model is None:
            st.error("Art model could not be loaded. Please ensure the model file is present.")
            return

        with st.spinner('🔍 Analyzing image with art classifier...'):
            art_label, art_conf = art_predict_from_pil(pil_img, art_model)

        # Display art classification result
        st.markdown("### 🎯 Art Classification Result")
        
        if art_label == 'ai_art':
            st.markdown(f"""
            <div class="result-success">
                <strong>Classification:</strong> AI-Generated Art<br>
                <strong>Confidence:</strong> {art_conf:.1%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-info">
                <strong>Classification:</strong> Human-Created Art<br>
                <strong>Confidence:</strong> {art_conf:.1%}
            </div>
            """, unsafe_allow_html=True)
        
        # Run artist classification if AI-generated
        if art_label == 'ai_art':
            st.markdown("---")
            
            artist_model, class_names, device = load_artist_model()
            if artist_model is None:
                st.error("Artist model could not be loaded. Please ensure the model file is present.")
                return

            with st.spinner('🎨 Identifying artistic style...'):
                top2 = artist_predict_topk(pil_img, artist_model, class_names, device, k=2)
            
            st.markdown("### 🎭 Top 2 Artistic Style Predictions")
            
            for idx, (name, conf) in enumerate(top2, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{idx}. {name}**")
                with col2:
                    st.metric(label="Confidence", value=f"{conf:.1%}")
        else:
            st.info('✓ Image identified as human-created art. Artist classification is only performed for AI-generated artwork.')
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# Main App Logic
# ============================================================================
def main():
    # Create sidebar navigation
    st.sidebar.markdown(f"""
    <div style="padding: 1rem 0; text-align: center; border-bottom: 2px solid {PRIMARY_COLOR}; margin-bottom: 1rem;">
        <h2 style="margin: 0; color: {PRIMARY_COLOR};">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select a page:",
        options=["🏠 Home", "🎨 Classifier"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About This App**
    
    This application combines two machine learning models to classify artwork:
    - Detection of AI-generated vs human-created art
    - Identification of artistic styles for AI-generated images
    """)
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "🎨 Classifier":
        classifier_page()

if __name__ == '__main__':
    main()