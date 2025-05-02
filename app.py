
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Configuración global
st.set_page_config(page_title="Pricing Simulator", page_icon="💰", layout="wide")
st.title("📊 Price Elasticity Simulator")

# Función para generar datos sintéticos más realistas
@st.cache_data
def generate_better_synthetic_data(seed=42):
    np.random.seed(seed)
    n_products = 10
    n_weeks = 52
    product_ids = [f"P{i+1}" for i in range(n_products)]
    weeks = range(1, n_weeks + 1)
    data = []

    category_price_ranges = {
        "A": (25, 40),
        "B": (40, 60),
        "C": (60, 85),
    }

    for pid in product_ids:
        category = np.random.choice(["A", "B", "C"])
        min_p, max_p = category_price_ranges[category]
        base_price = np.random.uniform(min_p, max_p)
        base_demand = np.random.randint(300, 1000)
        elasticity = np.random.uniform(-2.2, -0.8)

        for week in weeks:
            promo = np.random.choice([0, 1], p=[0.7, 0.3])
            price_variation = np.random.normal(0, 0.05)
            price = base_price * (1 + price_variation)
            if promo:
                price *= 0.9
            demand = base_demand * (price / base_price) ** elasticity
            demand *= (1.2 if promo else 1.0)
            units_sold = int(np.random.normal(demand, demand * 0.1))
            data.append([pid, category, week, round(price, 2), units_sold, promo])

    df = pd.DataFrame(data, columns=["product_id", "category", "week", "price", "units_sold", "promotion_flag"])
    return df

# Botón para regenerar datos
if "seed" not in st.session_state:
    st.session_state.seed = 42
if st.sidebar.button("🔄 Generate New Data"):
    st.session_state.seed += 1

df = generate_better_synthetic_data(seed=st.session_state.seed)

# Sidebar: configuración de simulación
st.sidebar.title("⚙️ Simulation Settings")
model_type = st.sidebar.selectbox("Select Model Type", ["Random Forest", "Linear Regression"])
categories = df["category"].unique().tolist()
categories.sort()
selected_cat = st.sidebar.selectbox("Select Category", categories)

# Entrenar modelos por categoría
models = {}
for cat in df["category"].unique():
    sub = df[df["category"] == cat]
    X = sub[["price", "promotion_flag"]]
    y = sub["units_sold"]
    linear = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    linear.fit(X, y)
    rf.fit(X, y)
    models[cat] = {"linear": linear, "rf": rf}

# Modelo seleccionado
model = models[selected_cat]["linear"] if model_type == "Linear Regression" else models[selected_cat]["rf"]

# Generar rango de precios realista para la categoría
cat_prices = df[df["category"] == selected_cat]["price"]
price_range = np.linspace(cat_prices.min(), cat_prices.max(), 100)

# Calcular curva sin promo
X_nopromo = pd.DataFrame({"price": price_range, "promotion_flag": [0]*len(price_range)})
units_nopromo = model.predict(X_nopromo)
revenue_nopromo = price_range * units_nopromo
opt_price_nopromo = float(price_range[np.argmax(revenue_nopromo)])
opt_revenue_nopromo = float(np.max(revenue_nopromo))

# Calcular curva con promo
X_promo = pd.DataFrame({"price": price_range * 0.9, "promotion_flag": [1]*len(price_range)})
units_promo = model.predict(X_promo)
revenue_promo = price_range * 0.9 * units_promo
opt_price_promo = float(price_range[np.argmax(revenue_promo)])
opt_revenue_promo = float(np.max(revenue_promo))

# Slider para cualquier precio manual
price_input = st.sidebar.slider(
    "Set Custom Price",
    float(cat_prices.min()),
    float(cat_prices.max()),
    float(opt_price_promo) if opt_revenue_promo > opt_revenue_nopromo else float(opt_price_nopromo)
)

# Predicción en el precio manual (sin y con promo)
X_manual_no = pd.DataFrame({"price": [price_input], "promotion_flag": [0]})
X_manual_yes = pd.DataFrame({"price": [price_input * 0.9], "promotion_flag": [1]})
revenue_manual_no = price_input * model.predict(X_manual_no)[0]
revenue_manual_yes = price_input * 0.9 * model.predict(X_manual_yes)[0]

st.markdown(f"### 🛒 Category: **{selected_cat}**")
st.markdown(f"**Model:** {model_type}")

# Mostrar resultados comparativos
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚫 No Promotion")
        st.markdown(f"**Optimal Price:** ${opt_price_nopromo:.2f}")
        st.markdown(f"**Estimated Revenue:** ${opt_revenue_nopromo:,.0f}")
    
    with col2:
        st.markdown("### ✅ With Promotion")
        st.markdown(f"**Optimal Price:** ${opt_price_promo:.2f}")
        st.markdown(f"**Real price after 10% promotion:** ${opt_price_promo*0.9:.2f}")
        st.markdown(f"**Estimated Revenue:** ${opt_revenue_promo:,.0f}")

    with col3:
        st.markdown("### 🧪 Custom Price")
        st.markdown(f"**Price:** ${price_input:.2f}")
        st.markdown(f"**Revenue (no promo):** ${revenue_manual_no:,.0f}")
        st.markdown(f"**Revenue (with promo):** ${revenue_manual_yes:,.0f}")

# Visualización de curvas
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Revenue Curve")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(price_range, revenue_nopromo, label="No Promotion", color="blue")
        ax.plot(price_range, revenue_promo, label="With 10% Promotion", color="green")
        ax.axvline(x=price_input, color="red", linestyle="--", label="Selected Price")
        ax.set_xlabel("Price")
        ax.set_ylabel("Estimated Revenue")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.markdown("#### 📉 Historical Sales Data")
        cat_data = df[df["category"] == selected_cat]
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=cat_data, x="price", y="units_sold", hue="promotion_flag", ax=ax1, palette="Set2", alpha=0.7)
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Units Sold")
        st.pyplot(fig1)
