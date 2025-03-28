import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Cache data loading and model training
@st.cache_data
def load_real_data():
    df = pd.read_csv('Manual-Data.csv')
    return df

@st.cache_data
def train_model(partner):
    df = load_real_data()
    partner_data = df[df['Partner'] == partner].copy()
    partner_data = partner_data[['Year', 'Exports']].dropna()
    
    # Convert exports to billions
    partner_data['Exports'] = partner_data['Exports'] / 1000
    
    return partner_data

# Streamlit app layout
st.title("🍁 Canada Trade Tariff Impact Simulator ESA 2025")

# Sidebar controls
with st.sidebar:
    st.header("Simulation Parameters")
    partner = st.selectbox(
        "Select Main Trading Partner",
        ["United States", "China"],
        index=0
    )
    tariff = st.slider(
        "Tariff Percentage",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
        help="Percentage tariff imposed by trading partner"
    )
    elasticity = st.slider(
        "Demand Elasticity",
        min_value=0.1,
        max_value=2.0,
        value=1.25,
        step=0.1,
        help="Price elasticity of demand for Canadian exports"
    )

# Load data and train model
data = train_model(partner)
if data.empty:
    st.error("No data available for selected partner")
    st.stop()

# Polynomial regression setup
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(data[['Year']])
model = LinearRegression()
model.fit(X_poly, data['Exports'])

# Generate forecast for next 5 years after last available data
last_year = data['Year'].max()
future_years = np.arange(last_year + 1, last_year + 6).reshape(-1, 1)
future_years_poly = poly.transform(future_years)
predicted = model.predict(future_years_poly)

# Calculate tariff impact using partial equilibrium model
price_increase = tariff / 100
demand_reduction = elasticity * price_increase
tariff_impact = predicted * (1 - demand_reduction)

# Create visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Year'], data['Exports'], 'b-o', label='Historical Exports')
ax.plot(future_years, predicted, 'g--', label='Baseline Forecast')
ax.plot(future_years, tariff_impact, 'r--', label='With Tariff Impact')
ax.fill_between(future_years.flatten(), predicted, tariff_impact, color='red', alpha=0.1)
ax.set_title(f"Impact of {tariff}% Tariff on Canadian Exports to {partner}")
ax.set_xlabel("Year")
ax.set_ylabel("Export Value (Billion CAD)")
ax.grid(True)
ax.legend()

# Display metrics
current_export = data[data['Year'] == last_year]['Exports'].values[0]
model_score = r2_score(data['Exports'], model.predict(X_poly))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("2023 Export Value", 
             f"${current_export:.1f}B",
             "Total exports")
with col2:
    st.metric("Model R² Score", 
             f"{model_score:.2f}",
             "Model Fit")
with col3:
    st.metric("Demand Reduction", 
             f"{demand_reduction*100:.1f}%",
             "From tariff")

st.pyplot(fig)

# Economic impact analysis
cumulative_loss = np.sum(predicted - tariff_impact)
jobs_at_risk = int(cumulative_loss * 6000)

st.subheader("📊 Economic Impact Assessment")
st.markdown(f"""
**Trade Impacts ({(last_year+1)}-{future_years[-1][0]}):**
- **Total export losses:** `${cumulative_loss:,.1f}B CAD`  
- **Percentage of expected trade:** `{(cumulative_loss/np.sum(predicted)*100):.1f}%`  

**Employment Impacts:**
- **Jobs at risk (5 years):** `{jobs_at_risk:,}` (est. 6,000 jobs per $1B exports)  
- **Annual average jobs at risk:** `{int(jobs_at_risk/5):,}`  

**Sector Sensitivity:**  
_Note:_ These estimates assume uniform impact across all sectors.  
Actual impacts may vary by industry (e.g., manufacturing vs agriculture).
""")
