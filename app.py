import streamlit as st
from model import train_model, predict, simulate_what_if
from data_fetch import get_onchain_data, get_social_momentum
from subscription import check_calls, process_payment, YOUR_WALLET
import matplotlib.pyplot as plt
import torch

# Load or train model
try:
    model = train_model()
    model.load_state_dict(torch.load('model.pth'))
except FileNotFoundError:
    model = train_model()

st.title("AI Token Longevity Predictor")
st.info("Note: 40% of subscription revenue goes to buyback & burn for $DEDU token!")

user_id = st.text_input("Your User ID (from subscription)")
token_address = st.text_input("Token Address")
token_name = st.text_input("Token Name (for social search)")

if st.button("Subscribe"):
    tier_options = {
        "Base (1 SOL - 1000 calls)": "base",
        "Addon 500 (0.3 SOL)": "addon_500",
        "Addon 200 (0.1 SOL)": "addon_200",
        "Addon 100 (0.05 SOL)": "addon_100"
    }
    tier_label = st.selectbox("Tier", list(tier_options.keys()))
    tier = tier_options[tier_label]
    st.write(f"Send payment to: {str(YOUR_WALLET)}")
    tx_sig = st.text_input("Enter TX Signature after payment")
    user_wallet = st.text_input("Your Wallet")
    if tx_sig and user_wallet:
        new_id = process_payment(tx_sig, user_wallet, tier)
        if new_id:
            st.success(f"Subscribed! User ID: {new_id}")
        else:
            st.error("Payment verification failed. Check TX.")

if st.button("Predict") and user_id:
    if not check_calls(user_id):
        st.error("No calls left or expired. Subscribe!")
    else:
        concentration, early_buys = get_onchain_data(token_address)
        social = get_social_momentum(token_name)
        features = [concentration, early_buys, social]
        
        prob, rug_risk, reward = predict(model, features)
        st.write(f"Survival Probability: {prob:.2%}")
        st.write(f"Rug Risk: {rug_risk:.2%}")
        st.write(f"Potential Reward (5x chance): {reward:.2f}")
        
        # What-if
        if st.checkbox("Run What-If Simulation"):
            multiplier = st.slider("Social Replies Multiplier", 1.0, 2.0)
            sim_prob, _, _ = simulate_what_if(model, features, {'social_momentum': multiplier})
            st.write(f"Projected Survival if Social Changes: {sim_prob:.2%}")
        
        # Chart
        fig, ax = plt.subplots()
        ax.bar(['Survival', 'Rug Risk'], [prob, rug_risk])
        st.pyplot(fig)
