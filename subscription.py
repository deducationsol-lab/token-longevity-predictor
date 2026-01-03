from solana.rpc.api import Client
from solana.publickey import PublicKey
import sqlite3
from datetime import datetime, timedelta
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
YOUR_WALLET = PublicKey("G2KVhsRMLdxfHYPd2aEUHL3T8EX4WgZSFW34q3JHF37o")
solana_client = Client(SOLANA_RPC_URL)

# Pricing tiers (in SOL)
BASE_SUB = 1.0  # 1000 calls
ADDON_500 = 0.3
ADDON_200 = 0.1
ADDON_100 = 0.05

conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                  (user_id TEXT PRIMARY KEY, wallet TEXT, calls_left INT, expiry DATE)''')
conn.commit()

def process_payment(tx_sig, user_wallet, tier='base'):
    try:
        tx = solana_client.get_transaction(tx_sig, max_supported_transaction_version=0)
        if not tx['result']:
            return False
        # Parse transfers (simplified; check for SOL transfer to YOUR_WALLET)
        pre_balances = tx['result']['meta']['preBalances']
        post_balances = tx['result']['meta']['postBalances']
        account_keys = tx['result']['transaction']['message']['accountKeys']
        wallet_index = next((i for i, key in enumerate(account_keys) if key == str(YOUR_WALLET)), None)
        if wallet_index is None:
            return False
        amount_received = (post_balances[wallet_index] - pre_balances[wallet_index]) / 1e9  # Lamports to SOL
        
        expected_amount = {
            'base': BASE_SUB,
            'addon_500': ADDON_500,
            'addon_200': ADDON_200,
            'addon_100': ADDON_100
        }.get(tier, 0)
        if amount_received != expected_amount:
            return False
        
        calls_to_add = {
            'base': 1000,
            'addon_500': 500,
            'addon_200': 200,
            'addon_100': 100
        }.get(tier, 0)
        expiry = (datetime.now() + timedelta(days=30)).date()
        
        user_id = str(uuid.uuid4())
        cursor.execute("INSERT OR REPLACE INTO users (user_id, wallet, calls_left, expiry) VALUES (?, ?, ?, ?)",
                       (user_id, user_wallet, calls_to_add, expiry))
        conn.commit()
        return user_id
    except Exception as e:
        print(f"Error processing payment: {e}")
        return False

def check_calls(user_id):
    cursor.execute("SELECT calls_left, expiry FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    if not row or datetime.now().date() > datetime.strptime(row[1], '%Y-%m-%d').date():
        return False
    if row[0] <= 0:
        return False
    cursor.execute("UPDATE users SET calls_left = calls_left - 1 WHERE user_id=?", (user_id,))
    conn.commit()
    return True
