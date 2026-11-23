import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Retail Dynamic Pricing", page_icon="🛍️", layout="wide")
st.title("🛍️ Dynamic Retail Pricing")
st.caption("Machine Learning + Demand-Based Pricing + Cryptography")

@st.cache_data
def load_data():
    return pd.read_csv("retail_price.csv")

@st.cache_resource
def train_model():
    data = load_data()
    target = "unit_price"
    # total_price is excluded because it is derived from qty × unit_price.
    # It would leak the target into the training data.
    features = [
        "product_id", "product_category_name", "qty", "freight_price",
        "product_name_lenght", "product_description_lenght",
        "product_photos_qty", "product_weight_g", "product_score",
        "customers", "weekday", "weekend", "holiday", "month", "year",
        "s", "volume", "comp_1", "ps1", "fp1", "comp_2", "ps2",
        "fp2", "comp_3", "ps3", "fp3", "lag_price"
    ]
    X = data[features].copy()
    y = data[target].copy()
    categorical = ["product_id", "product_category_name"]
    numeric = [c for c in features if c not in categorical]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ])
    model = Pipeline([
        ("prep", prep),
        ("rf", RandomForestRegressor(n_estimators=250, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    }
    return model, data, features, metrics

try:
    model, data, features, metrics = train_model()
except Exception as e:
    st.error(f"Could not load/train the model: {e}")
    st.stop()

with st.sidebar:
    st.header("Dataset")
    st.write(f"Records: **{len(data)}**")
    st.write(f"Products: **{data['product_id'].nunique()}**")
    st.write(f"Categories: **{data['product_category_name'].nunique()}**")
    st.write("Target: **unit_price**")

tab1, tab2, tab3 = st.tabs(["Price Prediction", "Model Performance", "Cryptography"])

with tab1:
    st.subheader("Predict a product's recommended price")
    product_ids = sorted(data["product_id"].unique())
    selected_id = st.selectbox("Select product", product_ids)
    row = data[data["product_id"] == selected_id].iloc[-1].copy()
    category = str(row["product_category_name"])
    st.write(f"Category: **{category}**")
    c1, c2, c3 = st.columns(3)
    qty = c1.number_input("Expected quantity sold", min_value=0, value=int(row["qty"]))
    customers = c2.number_input("Expected customers", min_value=1, value=max(1, int(row["customers"])))
    competitor = c3.number_input("Competitor 1 price", min_value=0.0, value=float(row["comp_1"]), step=1.0)
    c4, c5, c6 = st.columns(3)
    competitor2 = c4.number_input("Competitor 2 price", min_value=0.0, value=float(row["comp_2"]), step=1.0)
    competitor3 = c5.number_input("Competitor 3 price", min_value=0.0, value=float(row["comp_3"]), step=1.0)
    rating = c6.number_input("Product score", min_value=0.0, max_value=5.0, value=float(row["product_score"]), step=0.1)
    st.caption("Other product and calendar features use the selected product's latest dataset record.")
    if st.button("Calculate dynamic price", type="primary"):
        sample = row[features].to_frame().T
        sample.loc[:, "qty"] = qty
        sample.loc[:, "customers"] = customers
        sample.loc[:, "comp_1"] = competitor
        sample.loc[:, "comp_2"] = competitor2
        sample.loc[:, "comp_3"] = competitor3
        sample.loc[:, "product_score"] = rating
        base_price = float(model.predict(sample[features])[0])

        # Demand signal: quantity sold per customer, normalized against dataset median.
        demand_ratio = qty / max(customers, 1)
        baseline = float((data["qty"] / data["customers"].clip(lower=1)).median())
        demand_factor = np.clip(demand_ratio / max(baseline, 1e-9), 0.85, 1.15)
        # Higher demand raises price modestly; low demand reduces it.
        multiplier = float(demand_factor)
        recommended = max(0.0, base_price * multiplier)

        st.success("Price calculated")
        a, b, c = st.columns(3)
        a.metric("ML predicted base price", f"₹{base_price:,.2f}")
        b.metric("Demand multiplier", f"{multiplier:.3f}x")
        c.metric("Recommended dynamic price", f"₹{recommended:,.2f}")
        st.info("This is a prototype pricing rule. The dataset has no explicit inventory/supply field, so demand is estimated using quantity sold per customer; it is not a verified supply-demand elasticity model.")

with tab2:
    st.subheader("Random Forest Regression performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.3f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    m3.metric("R² score", f"{metrics['R2']:.3f}")
    st.write("Model: Random Forest Regressor")
    st.write("Train/test split: 80% / 20%")
    st.write("Target: unit_price")
    st.write("Features include product, sales, customer, calendar and competitor information.")
    st.dataframe(data.head(10), use_container_width=True)

with tab3:
    st.subheader("Secure the recommended price")
    st.write("AES-256-GCM encrypts the pricing record and authenticates it. SHA-256 provides a fingerprint for integrity demonstration.")
    if "aes_key" not in st.session_state:
        st.session_state.aes_key = AESGCM.generate_key(bit_length=256)
    price_to_secure = st.number_input("Price to encrypt (₹)", min_value=0.0, value=249.99, step=1.0, key="secure_price")
    if st.button("Encrypt price"):
        record = {"recommended_price": round(price_to_secure, 2), "currency": "INR"}
        plaintext = json.dumps(record, sort_keys=True).encode()
        nonce = os.urandom(12)
        aes = AESGCM(st.session_state.aes_key)
        ciphertext = aes.encrypt(nonce, plaintext, None)
        digest = hashlib.sha256(plaintext).hexdigest()
        st.session_state.secure_payload = {
            "nonce": nonce.hex(), "ciphertext": ciphertext.hex(),
            "sha256": digest
        }
    if "secure_payload" in st.session_state:
        payload = st.session_state.secure_payload
        st.code(json.dumps(payload, indent=2), language="json")
        if st.button("Decrypt and verify"):
            try:
                aes = AESGCM(st.session_state.aes_key)
                decrypted = aes.decrypt(bytes.fromhex(payload["nonce"]),
                                         bytes.fromhex(payload["ciphertext"]), None)
                actual_hash = hashlib.sha256(decrypted).hexdigest()
                st.success("AES-GCM authentication passed. Data decrypted successfully.")
                st.write("Decrypted record:")
                st.json(json.loads(decrypted))
                st.write("SHA-256 fingerprint matches:", actual_hash == payload["sha256"])
            except Exception:
                st.error("Authentication failed: encrypted data may have been modified or the key is incorrect.")
    st.warning("Educational demo: the AES key is held in the current Streamlit session. For production, use a secure key-management service; do not store keys beside ciphertext.")

st.divider()
st.caption("Academic prototype | Dynamic Retail Pricing using ML and Cryptography")
