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

ART_MODEL_PATH = 'DenseNet201_25_0.0001.h5'
ARTIST_MODEL_PATH = 'swin_transformer_artist_model_epoch50_lr1e-04_acc_65.12.pth'

art_labels = ["ai_art", "human_art"]

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
    return tf.keras.models.load_model(ART_MODEL_PATH)

@st.cache_resource
def load_artist_model():
    checkpoint = torch.load(ARTIST_MODEL_PATH, map_location=torch.device('cpu'))
    class_names = checkpoint['classes']
    base_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
    artist_model = CustomSwinModel(base_model, num_ftrs=base_model.head.in_features, num_classes=len(class_names))
    artist_model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artist_model = artist_model.to(device)
    artist_model.eval()
    return artist_model, class_names, device

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

def main():
    st.title('Art / Artist Classification')
    st.write('Upload an image to run the art classifier. If identified as AI-generated, the artist classifier will run and show top-2 predictions.')

    uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        st.image(pil_img, caption='Uploaded image', use_column_width=True)

        with st.spinner('Loading art model...'):
            art_model = load_art_model()

        art_label, art_conf = art_predict_from_pil(pil_img, art_model)
        st.write(f'**Art classification:** {art_label}  —  confidence: {art_conf:.3f}')

        if art_label == 'ai_art':
            with st.spinner('Running artist classifier...'):
                artist_model, class_names, device = load_artist_model()
                top2 = artist_predict_topk(pil_img, artist_model, class_names, device, k=2)
            st.write('**Top 2 artist predictions:**')
            for name, conf in top2:
                st.write(f'- {name}: {conf:.3f}')
        else:
            st.info('Image not identified as AI-generated; skipping artist classification.')

if __name__ == '__main__':
    main()