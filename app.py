import streamlit as st
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Must be the first Streamlit command
# Keep only this one at the top
st.set_page_config(page_title="Agribot", page_icon="🌾", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #37474f;
        margin-top: 1rem;
    }
    .input-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-section {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🌾 Crop Yield Prediction using Machine Learning</p>', unsafe_allow_html=True)


# Sidebar for app information
with st.sidebar:
    st.image("https://th.bing.com/th/id/OIP.PNDSOK2AwapHsq9yyqaa0QHaEo?rs=1&pid=ImgDetMain", width=150)
    st.markdown("### About This App")
    st.write("""
    This application uses machine learning to recommend the most suitable crop 
    based on soil and environmental parameters.
    
    Enter the required values and click 'Analyze' to get a crop recommendation.
    """)
    
    st.markdown("### How It Works")
    st.write("""
    The app uses a Random Forest model trained on a dataset of various soil parameters 
    and their corresponding optimal crops.
    """)

# Function to load model
@st.cache_resource
def load_model():
    try:
        with open('multi_target_forest.pkl', 'rb') as file:
            return pkl.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'multi_target_forest.pkl' exists in the app directory.")
        return None

# Load the model
model = load_model()

# Main input section
st.markdown('<p class="sub-header">Enter Soil and Environmental Parameters</p>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        N = st.number_input("Nitrogen (N) Level (kg/ha)", 
                           min_value=0, max_value=300, value=50, 
                           help="Amount of Nitrogen in the soil")
    
    with col2:
        P = st.number_input("Phosphorus (P) Level (kg/ha)", 
                           min_value=0, max_value=300, value=50, 
                           help="Amount of Phosphorus in the soil")
    
    with col3:
        K = st.number_input("Potassium (K) Level (kg/ha)", 
                           min_value=0, max_value=300, value=50, 
                           help="Amount of Potassium in the soil")
    
    with col1:
        temperature = st.slider('Temperature (°C)', 
                               min_value=0.0, max_value=45.0, value=25.0, step=0.1,
                               help="Average temperature in Celsius")
    
    with col2:
        humidity = st.slider("Humidity (%)", 
                            min_value=0, max_value=100, value=50,
                            help="Relative humidity percentage")
    
    with col3:
        ph = st.slider("pH Level", 
                      min_value=0.0, max_value=14.0, value=7.0, step=0.1,
                      help="pH level of the soil (0-14)")
    
    with col1:
        rainfall = st.slider("Rainfall (mm)", 
                            min_value=0, max_value=3000, value=200,
                            help="Annual rainfall in millimeters")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Corrected crop names list
crop_names = [
    "Grapes", "Muskmelon", "Coconut", "Chickpea", "Watermelon", "Papaya", "Pomegranate", 
    "Pigeonpeas", "Apple", "Maize", "Blackgram", "Cotton", "Mothbeans", "Coffee", 
    "Jute", "Orange", "Rice", "Banana", "Mungbean", "Kidneybeans", "Mango", "Lentil"
]

# Function to make prediction
def predict_crop(input_data):
    try:
        if model is None:
            return None
            
        # Reshape user input
        input_reshaped = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_reshaped)
        
        # Get the index of the highest probability
        crop_index = np.argmax(prediction)
        
        # Get probability score (confidence)
        probability = prediction[0][crop_index]
        
        return crop_names[crop_index], probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Create a single column for the analyze button
analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

# Display results
if analyze_button:
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    
    # Show a spinner while calculating
    with st.spinner('Analyzing soil parameters...'):
        user_input = [N, P, K, temperature, humidity, ph, rainfall]
        recommended_crop, confidence = predict_crop(user_input)
    
    if recommended_crop:
        st.markdown(f"### 🌱 Recommended Crop: **{recommended_crop}**")
        
        # Display confidence
        if confidence:
            confidence_percentage = confidence * 100
            st.progress(confidence_percentage/100)
            st.write(f"Confidence: {confidence_percentage:.2f}%")
        
        # Display some information about the recommended crop
        crop_info = {
             'Rice': "Requires high temperature (20-35°C), high humidity, and rainfall of 100-200cm. Grows best in alluvial soil with pH 5.5-6.5.",
        'Wheat': "Grows in cool climate with moderate rainfall (75-100cm). Prefers loamy soil with pH 6.0-7.0.",
        'Maize': "Requires warm climate, moderate rainfall, and well-drained soil with pH 5.5-7.0.",
        'Cotton': "Needs warm climate, moderate rainfall, and black soil with pH 6.0-7.5.",
        'Sugarcane': "Requires tropical climate, high temperature (21-27°C), high humidity, and rainfall of 75-150cm.",
        'Barley': "Grows in cool, dry climate and can tolerate various soil types with pH 6.0-8.5.",
        'Groundnut': "Requires warm climate, moderate rainfall, and sandy-loam soil with pH 6.0-6.5.",
        'Soybean': "Grows best in warm climate, moderate rainfall, and well-drained loamy soil with pH 6.0-7.5.",
        'Pulses': "Different varieties adapt to various climatic conditions, generally preferring well-drained soil.",
        'Mustard': "Requires cool climate, moderate rainfall, and loamy soil with good drainage.",
        'Potato': "Grows best in cool climate with well-drained, loose soil rich in organic matter.",
        'Tomato': "Requires warm climate, moderate moisture, and well-drained soil with pH 6.0-7.0.",
        'Onion': "Grows in mild climate without extreme heat or cold, prefers well-drained soil with pH 6.0-7.0.",
        'Chilies': "Requires warm climate, moderate rainfall, and well-drained soil rich in organic matter.",
        'Green Peas': "Grows best in cool climate with well-drained soil and pH 6.0-7.5.",
        'Mango': "Requires tropical climate, moderate rainfall, and deep soil with good drainage.",
        'Papaya': "Grows in tropical and subtropical climates, requires well-drained soil rich in organic matter."
        }
        
        if recommended_crop in crop_info:
            st.info(f"**About {recommended_crop}**: {crop_info[recommended_crop]}")
        
        # Display the input parameters
        with st.expander("View Input Parameters"):
            parameter_df = pd.DataFrame({
                'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                             'Temperature', 'Humidity', 'pH', 'Rainfall'],
                'Value': user_input,
                'Unit': ['kg/ha', 'kg/ha', 'kg/ha', '°C', '%', 'pH', 'mm']
            })
            st.dataframe(parameter_df, use_container_width=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<p class="footer">Developed for Agricultural Planning and Crop Selection</p>', unsafe_allow_html=True)

# Load the pre-trained model
with open('crop_production_model.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

# Streamlit application

    st.title("Crop Production ")
    
    # Instructions
    st.write("Please input the following details to predict the crop production.")

    # User input for 'Season'
    season = st.selectbox(
        'Select the season:',
        ['Kharif', 'Zaid', 'Rabi', 'Summer', 'Winter']
    )
    
    # User input for 'Crop'
    crop = st.selectbox(
        'Select the crop:',
        ['Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Black pepper', 'Blackgram', 'Brinjal', 
    'Castor seed', 'Cabbage', 'Cashewnut', 'Coconut', 'Cotton(lint)', 'Coriander', 'Cowpea(Lobia)', 
    'Dry chillies', 'Dry ginger', 'Ginger', 'Gram', 'Garlic', 'Groundnut', 'Guar seed', 'Jowar', 'Khesari', 
    'Korra', 'Lemon', 'Linseed', 'Mango', 'Masoor', 'Mesta', 'Maize', 'Moth', 'Moong(Green Gram)', 'Niger seed', 
    'Onion', 'Orange', 'Other Cereals & Millets', 'Other  Rabi pulses', 'Paddy', 'Papaya', 'Peas & beans (Pulses)', 
    'Pineapple', 'Potato', 'Pome Granet', 'Rapeseed &Mustard', 'Rice', 'Ragi', 'Safflower', 'Sesamum', 'Small millets', 
    'Soyabean', 'Sunflower', 'Sweet potato', 'Tapioca', 'Tomato', 'Tobacco', 'Turmeric', 'Urad', 'Varagu', 'Wheat']
    )

    # User input for 'Area'
    area = st.number_input('Enter the area of Farm:', min_value=0.0)

    # Button to make the prediction
    if st.button('Analyze Production'):
        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[season, crop, area]], columns=['Season', 'Crop', 'Area'])
        
        # Make the prediction using the model
        prediction = model.predict(input_data)
        
        # Show the predicted result
        st.write(f"Predicted Crop Production: {prediction[0]:.2f} units")





######      AGRIBOT.PY




# Download NLTK resources
@st.cache_resource

def get_tfidf_matrix(chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(chunks)

# Example process using threading
def process_pdf_in_background(pdf_file_path):
    result = st.session_state.chatbot.process_pdf(pdf_file_path)
    st.session_state.pdf_info = result
    st.session_state.pdf_uploaded = True

import threading

# Define pdf_file_path
pdf_file_path = r"C:\Users\Suhas\OneDrive\Desktop\Crop_Yeild_Prediction_Using_ML-main\AGRICULTURE.pdf"  # Adjust path if needed

# Running processing in background
if pdf_file_path:
    thread = threading.Thread(target=process_pdf_in_background, args=(pdf_file_path,))
    thread.start()
    st.spinner("Processing in background...")

def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_resources()

class CropYieldChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.pdf_content = []
        self.pdf_chunks = []
        self.tfidf_matrix = None
        self.pdf_loaded = False

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return "".join([page.extract_text() for page in pdf_reader.pages]), len(pdf_reader.pages)
        except Exception as e:
            return f"Error: {str(e)}", 0

    def process_pdf(self, pdf_file_path):
        """Process PDF and prepare for question answering"""
        full_text, num_pages = self.extract_text_from_pdf(pdf_file_path)
        if full_text.startswith("Error"):
            return full_text
        
        self.pdf_content = full_text
        self.pdf_chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', full_text) if len(chunk.strip()) > 50]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.pdf_chunks)
        self.pdf_loaded = True

        return {'status': 'success', 'pages': num_pages, 'chunks': len(self.pdf_chunks), 'topics': self.extract_key_topics()}

    def extract_key_topics(self, num_topics=5):
        """Extract crop-related topics from the PDF content"""
        crop_terms = ['rice', 'wheat', 'corn', 'maize', 'soybean', 'potato', 'yield', 'harvest', 'irrigation', 'fertilizer', 'soil']
        return [term for term in crop_terms if term.lower() in self.pdf_content.lower()]

    def find_best_chunk(self, query, top_n=3):
        """Find most relevant chunks for a query"""
        if not self.pdf_chunks:
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return [(self.pdf_chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0.05]

    def answer_query(self, query):
        """Answer a query using PDF content"""
        if not self.pdf_loaded:
            return "Please upload a document first."
        relevant_chunks = self.find_best_chunk(query)
        if not relevant_chunks:
            return "No relevant information found."
        return "\n\n".join([f"{chunk[:300]}... (Confidence: {'🟢 High' if similarity > 0.3 else '🟡 Medium'})" for chunk, similarity in relevant_chunks])

def main():

    # Initialize session state variables
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CropYieldChatbot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    
    st.title("🌾 Agribot")

    # Direct PDF path or file uploader
    pdf_file_path = r"C:\Users\Suhas\OneDrive\Desktop\Crop_Yeild_Prediction_Using_ML-main\AGRICULTURE.pdf"  #pdf path
    if pdf_file_path:
        with st.spinner('Processing ...'):
            result = st.session_state.chatbot.process_pdf(pdf_file_path)
            if isinstance(result, str) and result.startswith("Error"):
                st.error(result)
            else:
                st.session_state.pdf_uploaded = True
                st.success("processed successfully!")

    # Chat interface
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Chat With Agribot")
        for question, answer in st.session_state.chat_history:
            st.markdown(f"**You asked:** {question}")
            st.markdown(f"**Response:** {answer}")
            st.markdown("---")
    
    with col2:
        st.subheader("Ask a Question")
        user_question = st.text_area("Enter your question:", disabled=not st.session_state.pdf_uploaded)
        if st.button("Ask", disabled=not st.session_state.pdf_uploaded) and user_question:
            answer = st.session_state.chatbot.answer_query(user_question)
            st.session_state.chat_history.append((user_question, answer))
            st.rerun()

        if st.session_state.pdf_uploaded:
            st.subheader("Example Questions")
            example_questions = [
                "What were the highest crop yields this year?",
                "How does rainfall affect crop production?",
                "What are the recommended fertilizers?",
                "Which crops perform best in drought conditions?"
            ]
            for question in example_questions:
                if st.button(question):
                    answer = st.session_state.chatbot.answer_query(question)
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()


if __name__ == "__main__":
    main()
