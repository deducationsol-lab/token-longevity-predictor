import requests
from solana.rpc.api import Client
from solders.pubkey import Pubkey as PublicKey  # Updated import
import tweepy
from textblob import TextBlob
import time
import os
from dotenv import load_dotenv

load_dotenv()

SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

solana_client = Client(SOLANA_RPC_URL)

def get_onchain_data(token_address):
    try:
        pubkey = PublicKey(token_address)
        # Get largest token holders (better for concentration)
        response = solana_client.get_token_largest_accounts(pubkey)
        if not response.value:
            return 0.5, 100
        
        top_holders = [acc.amount for acc in response.value[:10]]
        total_top = sum(top_holders)
        # Rough concentration score (higher = more concentrated/top-heavy)
        concentration = total_top / (10 ** 9) if top_holders else 0.5
        
        # Recent signatures as proxy for early/activity
        sigs = solana_client.get_signatures_for_address(pubkey, limit=50)
        recent_count = len([s for s in sigs.value if s.block_time and time.time() - s.block_time < 3600 * 24])
        return concentration, recent_count
    except Exception as e:
        print(f"On-chain error: {e}")
        return 0.5, 100  # Safe fallback

def get_social_momentum(token_name):
    if not X_BEARER_TOKEN:
        return 50  # Fallback
    try:
        client = tweepy.Client(bearer_token=X_BEARER_TOKEN)
        query = f"{token_name} (pump OR pumpfun OR solana) lang:en -is:retweet"
        tweets = client.search_recent_tweets(query=query, max_results=20)
        if not tweets.data:
            return 10
        sentiments = [TextBlob(t.text).sentiment.polarity for t in tweets.data]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        momentum = len(tweets.data) * (avg_sentiment + 1)
        return momentum
    except Exception as e:
        print(f"Social error: {e}")
        return 20
