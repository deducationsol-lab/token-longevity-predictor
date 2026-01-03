from solana.rpc.api import Client
from solders.pubkey import Pubkey
import sqlite3
from datetime import datetime, timedelta
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
YOUR_WALLET_STR = "G2KVhsRMLdxfHYPd2aEUHL3T8EX4WgZSFW34q3JHF37o"
YOUR_WALLET = Pubkey.from_string(YOUR_WALLET_STR)
solana_client = Client(SOLANA_RPC_URL)

# Tiers in SOL
TIERS = {
    "base": (1.0, 1000),
    "addon_500": (0.3, 500),
    "addon_200": (0.1, 200),
    "addon_100": (0.05, 100)
}

conn = sqlite3.connect('database.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users 
             (user_id TEXT PRIMARY KEY, wallet TEXT, calls_left INTEGER, expiry TEXT)''')
conn.commit()

def process_payment(tx_sig, user_wallet, tier):
    try:
        tx = solana_client.get_transaction(tx_sig, max_supported_transaction_version=0)
        if not tx.get('result'):
            return False
        
        meta = tx['result']['meta']
        pre = meta['preBalances']
        post = meta['postBalances']
        keys = tx['result']['transaction']['message']['accountKeys']
        
        try:
            wallet_idx = keys.index(str(YOUR_WALLET))
        except ValueError:
            return False
        
        received = (post[wallet_idx] - pre[wallet_idx]) / 1e9
        expected_amount, calls = TIERS.get(tier, (0, 0))
        
        if abs(received - expected_amount) < 0.01:  # small tolerance
            expiry = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            user_id = str(uuid.uuid4())
            c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?)",
                      (user_id, user_wallet, calls, expiry))
            conn.commit()
            return user_id
        return False
    except:
        return False

def check_calls(user_id):
    c.execute("SELECT calls_left, expiry FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if not row:
        return False
    calls_left, expiry = row
    if datetime.strptime(expiry, '%Y-%m-%d') < datetime.now() or calls_left <= 0:
        return False
    c.execute("UPDATE users SET calls_left = calls_left - 1 WHERE user_id=?", (user_id,))
    conn.commit()
    return True
