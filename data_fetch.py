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
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

solana_client = Client(SOLANA_RPC_URL)

def get_onchain_data(token_address):
    try:
        pubkey = PublicKey(token_address)
        # Get token holders (simplified)
        response = solana_client.get_token_largest_accounts(pubkey)
        if not response.value:
            return 0.5, 100
        
        top_holders = [acc.amount for acc in response.value[:10]]
        total_top = sum(top_holders)
        concentration = total_top / (10 ** 9) if top_holders else 0.5  # rough estimate
        
        # Recent transactions (proxy for early buys)
        sigs = solana_client.get_signatures_for_address(pubkey, limit=50)
        recent_count = len([s for s in sigs.value if time.time() - s.block_time < 3600 * 24])
        return concentration, recent_count
    except:
        return 0.5, 100  # fallback

def get_social_momentum(token_name):
    if not X_BEARER_TOKEN:
        return 50  # fallback if no key
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
    except:
        return 20
