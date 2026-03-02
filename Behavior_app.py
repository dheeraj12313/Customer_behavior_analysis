import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Data & Model Caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('clean_train.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'clean_train.csv' not found. Please ensure it is in the same directory as this script.")
        return pd.DataFrame()

@st.cache_resource
def train_model(df):
    if df.empty:
        return None
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Explore the data or make a prediction.")
app_mode = st.sidebar.radio("Choose a section:", ["Project Overview & EDA", "Predict Retention"])

st.sidebar.markdown("---")
st.sidebar.info("This application predicts the likelihood of an employee looking for a new job based on their demographics and experience.")

# 4. App Sections
if app_mode == "Project Overview & EDA":
    st.title("📊 Employee Retention Analytics Dashboard")
    st.markdown("Welcome to the Employee Retention Prediction tool. This dashboard explores the factors that influence an employee's decision to leave a company.")
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Employee Records", len(df))
        col2.metric("Features Analyzed", len(df.columns) - 1)
        col3.metric("Overall Turnover Rate", f"{(df['target'].mean() * 100):.1f}%")
        
        st.markdown("---")
        st.subheader("Data Insights")
        
        tab1, tab2 = st.tabs(["Target Distribution", "Experience vs Training Hours"])
        
        with tab1:
            target_counts = df['target'].value_counts().reset_index()
            target_counts.columns = ['Status', 'Count']
            target_counts['Status'] = target_counts['Status'].map({0: 'Staying', 1: 'Leaving'})
            fig_target = px.bar(target_counts, x='Status', y='Count', color='Status', 
                                title="Distribution of Employee Status",
                                color_discrete_sequence=['#2ecc71', '#e74c3c'])
            st.plotly_chart(fig_target, use_container_width=True)
            
        with tab2:
            fig_scatter = px.scatter(df, x='experience', y='training_hours', color='target',
                                     title="Experience vs. Training Hours (Colored by Target)",
                                     opacity=0.6, color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        if st.checkbox("Show Raw Data"):
            st.dataframe(df.head(100))

elif app_mode == "Predict Retention":
    st.title("🔮 Predict Employee Retention")
    st.markdown("Enter the employee's details below to predict if they are likely to look for a new job.")
    
    if model is not None and not df.empty:
        
        # --- DICTIONARIES FOR UI MAPPING ---
        # Note: These map the readable text back to the numbers your model expects.
        # If your specific dataset encoded things slightly differently (e.g., Male was 0 instead of 1), 
        # you can just swap the numbers in these lists!
        mappings = {
            'gender': {'Female': 0, 'Male': 1, 'Other': 2},
            'relevent_experience': {'No relevant experience': 0, 'Has relevant experience': 1},
            'enrolled_university': {'No enrollment': 0, 'Full time course': 1, 'Part time course': 2},
            'education_level': {'Primary School': 0, 'High School': 1, 'Graduate': 2, 'Masters': 3, 'Phd': 4},
            'major_discipline': {'STEM': 0, 'Business Degree': 1, 'Arts': 2, 'Humanities': 3, 'No Major': 4, 'Other': 5},
            'company_type': {'Pvt Ltd': 0, 'Funded Startup': 1, 'Early Stage Startup': 2, 'Other': 3, 'Public Sector': 4, 'NGO': 5},
            'company_size': {'<10': 0, '10-49': 1, '50-99': 2, '100-500': 3, '500-999': 4, '1000-4999': 5, '5000-9999': 6, '10000+': 7}
        }
        
        with st.form("prediction_form"):
            st.subheader("Employee Profile")
            
            input_data = {}
            col1, col2, col3 = st.columns(3)
            
            features = [col for col in df.columns if col != 'target']
            
            for i, feature in enumerate(features):
                # Distribute inputs across 3 columns
                if i % 3 == 0: current_col = col1
                elif i % 3 == 1: current_col = col2
                else: current_col = col3
                
                with current_col:
                    # Clean up the display name (e.g., 'relevent_experience' -> 'Relevent Experience')
                    display_name = feature.replace('_', ' ').title()
                    
                    if feature in mappings:
                        # If it's a categorical feature, show the text options
                        options = list(mappings[feature].keys())
                        selected_text = st.selectbox(display_name, options=options)
                        # Save the corresponding number for the model
                        input_data[feature] = mappings[feature][selected_text]
                    else:
                        # If it's a numerical feature (like city, training_hours, experience)
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        mean_val = float(df[feature].mean())
                        input_data[feature] = st.number_input(display_name, min_value=min_val, max_value=max_val, value=mean_val)
            
            submit_button = st.form_submit_button(label="Predict Retention Status")
            
        if submit_button:
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            if prediction == 1:
                st.error(f"⚠️ High Risk of Departure")
                st.markdown(f"**Probability of leaving:** {probability * 100:.1f}%")
                st.progress(float(probability))
            else:
                st.success(f"✅ Likely to Stay")
                st.markdown(f"**Probability of leaving:** {probability * 100:.1f}%")
                st.progress(float(probability))
    else:
        st.warning("Model is not ready. Please check your data files.")