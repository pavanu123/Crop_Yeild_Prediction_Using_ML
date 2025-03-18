import streamlit as st
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import os

# Page configuration
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

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

# Create columns for buttons
button_col1, button_col2 = st.columns([1, 1])

with button_col1:
    analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

with button_col2:
    accuracy_button = st.button("📊 Show Model Accuracy", use_container_width=True)

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
        'Papaya': "Grows in tropical and subtropical climates, requires well-drained soil rich in organic matter."

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

# Calculate and show model accuracy if requested
if accuracy_button and model is not None:
    try:
        # Path to test data - using a relative path to make it more portable
        data_path = "crop_production.csv"
        
        if not os.path.exists(data_path):
            st.warning(f"Test data file not found at {data_path}. Please upload the file or provide the correct path.")
            
            # Allow user to upload a CSV file
            uploaded_file = st.file_uploader("Upload crop_production.csv", type="csv")
            if uploaded_file is not None:
                test_data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            else:
                st.stop()
        else:
            test_data = pd.read_csv(data_path)
        
        columns_to_consider = ['N', 'P', 'K', 'temperature','humidity', 'ph','rainfall']
        # Prepare data
        X = test_data[columns_to_consider]
        y = test_data['Crop'].values.reshape(-1, 1)
        
        # Apply OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        y_one_hot = encoder.fit_transform(y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Convert predictions and true values to class labels
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_one_hot, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true_class, y_pred_class)
        
        # Display accuracy with visualization
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Model Performance")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
            
        with col2:
            # Create a progress bar for accuracy
            st.progress(accuracy)
            st.caption("Model accuracy based on test data")
        
        # Show sample predictions
        with st.expander("View Sample Predictions"):
            sample_size = min(5, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            
            sample_X = X.iloc[sample_indices]
            sample_y_true = [crop_names[idx] for idx in y_true_class[sample_indices]]
            sample_y_pred = [crop_names[idx] for idx in y_pred_class[sample_indices]]
            
            sample_df = pd.DataFrame({
                'True Crop': sample_y_true,
                'Predicted Crop': sample_y_pred,
                'Correct Prediction': [t == p for t, p in zip(sample_y_true, sample_y_pred)]
            })
            
            st.dataframe(sample_df, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error calculating accuracy: {str(e)}")

# Footer
st.markdown('<p class="footer">Developed for Agricultural Planning and Crop Selection</p>', unsafe_allow_html=True)




# Load the pre-trained model
with open('crop_production_model.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

# Streamlit application
def main():
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
    'Soyabean', 'Sunflower', 'Sweet potato', 'Tapioca', 'Tomato', 'Tobacco', 'Turmeric', 'Urad', 'Varagu', 'Wheat']  # Add all crops you have
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

if __name__ == "__main__":
    main()
