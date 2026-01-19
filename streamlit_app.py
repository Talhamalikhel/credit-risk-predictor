import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# Define the model architecture (same as training)
class CreditRiskNN(nn.Module):
    def __init__(self, input_dim):
        super(CreditRiskNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Load model and scaler
@st.cache_resource
def load_model():
    model = CreditRiskNN(input_dim=23)
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏦 Credit Risk Prediction System")
st.markdown("---")

# Create tabs for better organization
tab1, tab2 = st.tabs(["📊 Make Prediction", "ℹ️ About"])

with tab1:
    # Basic Information Section
    st.subheader("📊 Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        limit_bal = st.number_input("Credit Limit (NT$)", min_value=0, value=200000, step=10000)
        sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female", index=1)
        education = st.selectbox("Education", options=[1, 2, 3, 4], 
                                format_func=lambda x: {1: "Graduate School", 2: "University", 
                                                    3: "High School", 4: "Others"}[x], index=1)
    
    with col2:
        marriage = st.selectbox("Marital Status", options=[1, 2, 3], 
                            format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x], index=0)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
    
    st.markdown("---")
    
    # Payment History Section
    st.subheader("💳 Payment History (Last 6 Months)")
    st.caption("0 = Paid on time, 1 = 1 month delay, 2 = 2 months delay, 3+ = 3+ months delay")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pay_0 = st.selectbox("September (Most Recent)", options=[0, 1, 2, 3], index=0, key="pay0")
        pay_2 = st.selectbox("August", options=[0, 1, 2, 3], index=0, key="pay2")
    
    with col2:
        pay_3 = st.selectbox("July", options=[0, 1, 2, 3], index=0, key="pay3")
        pay_4 = st.selectbox("June", options=[0, 1, 2, 3], index=0, key="pay4")
    
    with col3:
        pay_5 = st.selectbox("May", options=[0, 1, 2, 3], index=0, key="pay5")
        pay_6 = st.selectbox("April", options=[0, 1, 2, 3], index=0, key="pay6")
    
    st.markdown("---")
    
    # Bill Amounts Section
    st.subheader("💰 Bill Amounts (NT$)")
    bill_cols = st.columns(3)
    
    with bill_cols[0]:
        bill_amt1 = st.number_input("September Bill", min_value=0, value=5000, step=1000)
        bill_amt2 = st.number_input("August Bill", min_value=0, value=4500, step=1000)
    
    with bill_cols[1]:
        bill_amt3 = st.number_input("July Bill", min_value=0, value=4000, step=1000)
        bill_amt4 = st.number_input("June Bill", min_value=0, value=3500, step=1000)
    
    with bill_cols[2]:
        bill_amt5 = st.number_input("May Bill", min_value=0, value=3000, step=1000)
        bill_amt6 = st.number_input("April Bill", min_value=0, value=2500, step=1000)
    
    st.markdown("---")
    
    # Payment Amounts Section
    st.subheader("💵 Payment Amounts (NT$)")
    pay_cols = st.columns(3)
    
    with pay_cols[0]:
        pay_amt1 = st.number_input("September Payment", min_value=0, value=2000, step=500)
        pay_amt2 = st.number_input("August Payment", min_value=0, value=2000, step=500)
    
    with pay_cols[1]:
        pay_amt3 = st.number_input("July Payment", min_value=0, value=1500, step=500)
        pay_amt4 = st.number_input("June Payment", min_value=0, value=1500, step=500)
    
    with pay_cols[2]:
        pay_amt5 = st.number_input("May Payment", min_value=0, value=1000, step=500)
        pay_amt6 = st.number_input("April Payment", min_value=0, value=1000, step=500)
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Analyze Credit Risk", type="primary"):
        # Prepare input data
        input_data = np.array([[
            limit_bal, sex, education, marriage, age,
            pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
            bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
            pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
        ]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            probability = model(input_tensor).item()
        
        # Display results
        st.markdown("### 📊 Prediction Results")
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High Risk"
            recommendation = "Deny Credit Application"
            color = "🔴"
            st.error(f"{color} **{risk_level}** - {recommendation}")
        elif probability > 0.5:
            risk_level = "Medium Risk"
            recommendation = "Review Manually or Offer Lower Limit"
            color = "🟡"
            st.warning(f"{color} **{risk_level}** - {recommendation}")
        else:
            risk_level = "Low Risk"
            recommendation = "Approve Credit Application"
            color = "🟢"
            st.success(f"{color} **{risk_level}** - {recommendation}")
        
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric(label="Default Probability", value=f"{probability*100:.2f}%")
        
        with col_m2:
            st.metric(label="Risk Level", value=risk_level)
        
        with col_m3:
            st.metric(label="Confidence", value=f"{abs(probability-0.5)*200:.1f}%")
        
        # Progress bar
        st.progress(probability)
        
        # Additional insights
        with st.expander("📈 Understanding the Prediction"):
            st.write(f"""
            **Probability Score:** {probability*100:.2f}%
            
            **Threshold:** 50% (above = higher risk)
            
            **Model Accuracy:** ~82% on test data
            
            **What this means:**
            {"This customer has a high likelihood of defaulting on their credit card payment next month. Consider denying the application or requesting additional security." if probability > 0.7 
            else "This customer shows moderate risk. Manual review is recommended before making a decision." if probability > 0.5
            else "This customer is likely to make their credit card payment on time. Low risk for default."}
            """)

with tab2:
    st.subheader("About This Application")
    st.write("""
    This Credit Risk Prediction System uses a **PyTorch Neural Network** trained on Taiwan credit card default data.
    
    **Features:**
    - Predicts probability of credit card payment default
    - 82% accuracy on test data
    - Real-time predictions
    - User-friendly interface
    
    **Model Architecture:**
    - Input Layer: 23 features
    - Hidden Layer 1: 64 neurons + ReLU + BatchNorm + Dropout
    - Hidden Layer 2: 32 neurons + ReLU + BatchNorm + Dropout
    - Output Layer: 1 neuron + Sigmoid activation
    
    **Dataset:**
    Taiwan credit card default dataset with 30,000 customers
    """)
    
    st.info("💡 **Tip:** Lower payment history values (paying on time) and consistent payments reduce default risk.")

# Footer
st.markdown("---")
st.caption("🔒 Model trained on Taiwan credit card default dataset | Built with PyTorch & Streamlit")