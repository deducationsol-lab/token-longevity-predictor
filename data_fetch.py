import requests
from solana.rpc.api import Client
from solana.publickey import PublicKey
import tweepy
from textblob import TextBlob
import time
import os
from dotenv import load_dotenv

load_dotenv()

SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
PUMP_FUN_API = "https://api.moralis.io/solana/pumpfun/tokens"  # Adjust if using different API
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

solana_client = Client(SOLANA_RPC_URL)

def get_onchain_data(token_address):
    try:
        pubkey = PublicKey(token_address)
        accounts = solana_client.get_token_accounts_by_owner(pubkey, {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"})
        balances = [acc['amount'] for acc in accounts.value]
        total_supply = sum(balances)
        if total_supply == 0:
            return 0, 0
        concentration = sum((b / total_supply) ** 2 for b in balances)
        signatures = solana_client.get_signatures_for_address(pubkey, limit=100)
        early_buys = len([sig for sig in signatures.value if time.time() - sig['blockTime'] < 3600])
        return concentration, early_buys
    except Exception as e:
        print(f"Error fetching on-chain data: {e}")
        return 0, 0

def get_social_momentum(token_name):
    try:
        client = tweepy.Client(bearer_token=X_BEARER_TOKEN)
        query = f"{token_name} lang:en -is:retweet"
        tweets = client.search_recent_tweets(query=query, max_results=50)
        if not tweets.data:
            return 0
        sentiments = [TextBlob(tweet.text).sentiment.polarity for tweet in tweets.data]
        avg_sentiment = sum(sentiments) / len(sentiments)
        return len(tweets.data) * (avg_sentiment + 1)
    except Exception as e:
        print(f"Error fetching social data: {e}")
        return 0

def get_pumpfun_token_info(token_address):
    try:
        headers = {"Authorization": os.getenv("PUMP_FUN_API_KEY")}
        response = requests.get(f"{PUMP_FUN_API}/{token_address}", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('market_cap', 0), data.get('migration_status', False)
        return 0, False
    except Exception as e:
        print(f"Error fetching Pump.fun info: {e}")
        return 0, False
