# AI-Powered Token Longevity Predictor

An app that uses machine learning to predict the survival probability of new Pump.fun token launches. Analyzes on-chain data (holder concentration, early buys) and social momentum to score risk/reward and simulate scenarios.

## Features
- Predict token longevity (e.g., chance to hit $1M MC or Raydium migration).
- Risk/reward scores and "what-if" simulations.
- Monthly subscriptions via Solana payments for API calls.
- Built with PyTorch, Solana API, X API, and Streamlit.

**Important Note**: 40% of all revenue from subscriptions will be used for buyback and burn of $DEDU token to support the community and token value.

## Setup
1. Install dependencies: `pip install solana tweepy torch requests pandas numpy textblob streamlit sqlite3 python-dotenv matplotlib`.
2. Create a `.env` file with your keys:
