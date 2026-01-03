import streamlit as st
from model import train_model, predict, simulate_what_if
from data_fetch import get_onchain_data, get_social_momentum
from subscription import check_calls, process_payment
import matplotlib.pyplot as plt
import torch
import os

# Load or train model
if os.path.exists('model.pth'):
    model = train_model()
    model.load_state_dict(torch.load('model.pth'))
else:
    model = train_model()

st.title("AI Token Longevity Predictor")
st.info("Note: 40% of all subscription revenue will be used for buyback & burn of $DEDU token!")

user_id = st.text_input("Your User ID (received after subscription)")

token_address = st.text_input("Pump.fun Token Address (mint)")
token_name = st.text_input("Token Name or Ticker (for social search, e.g. DEDU)")

if st.button("Subscribe / Add Calls"):
    st.write("### Subscription Tiers")
    st.write("- **Base**: 1 SOL → 1,000 calls/month")
    st.write("- **Addon 500**: 0.3 SOL → +500 calls")
    st.write("- **Addon 200**: 0.1 SOL → +200 calls")
    st.write("- **Addon 100**: 0.05 SOL → +100 calls")
    st.write("**Send exact amount to:** `G2KVhsRMLdxfHYPd2aEUHL3T8EX4WgZSFW34q3JHF37o`")
    
    tier = st.selectbox("Choose tier", ["base", "addon_500", "addon_200", "addon_100"])
    tx_sig = st.text_input("Paste Transaction Signature here")
    user_wallet = st.text_input("Your Solana Wallet Address")
    
    if st.button("Verify Payment"):
        new_id = process_payment(tx_sig, user_wallet, tier)
        if new_id:
            st.success(f"Subscription active! Your User ID: `{new_id}`")
            st.info("Save this User ID — you'll need it to make predictions.")
        else:
            st.error("Payment not verified. Check amount, wallet, and try again.")

if st.button("Run Prediction") and user_id:
    if not check_calls(user_id):
        st.error("No calls remaining or subscription expired. Please subscribe again.")
    else:
        with st.spinner("Fetching data and predicting..."):
            concentration, early_buys = get_onchain_data(token_address)
            social = get_social_momentum(token_name)
            features = [concentration, early_buys, social]
            
            prob, rug_risk, reward = predict(model, features)
            
            st.success("Prediction Complete!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Survival Probability", f"{prob:.1%}")
            col2.metric("Rug Risk", f"{rug_risk:.1%}")
            col3.metric("Potential Reward Score", f"{reward:.1f}x")
            
            # What-if simulation
            st.write("### What-If Simulation")
            multiplier = st.slider("Increase social momentum by", 1.0, 3.0, 2.0)
            sim_prob, sim_rug, sim_reward = simulate_what_if(model, features, {'social_momentum': multiplier})
            st.write(f"If social momentum ×{multiplier}: Survival → **{sim_prob:.1%}** | Reward → **{sim_reward:.1f}x**")
            
            # Simple chart
            fig, ax = plt.subplots()
            ax.bar(['Survival Chance', 'Rug Risk'], [prob, rug_risk], color=['#00ff00', '#ff4444'])
            ax.set_ylim(0, 1)
            st.pyplot(fig)
