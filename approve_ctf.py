#!/usr/bin/env python3
"""
Approve USDC (config.USDC_ADDRESS) to the CTF contract for merge/redeem flows.

Spender (CTF): uses config.CTF_CONTRACT_ADDRESS
Token: config.USDC_ADDRESS (currently the native USDC 0x3c499c... on Polygon)

Usage:
  source venv/bin/activate
  python3 approve_ctf.py

Requires env/config:
  RPC_URL / ALCHEMY_RPC_URL
  PRIVATE_KEY
  POLYGON_ADDRESS (for logging)
  CHAIN_ID (defaults to 137)
"""

import json
import os
from dotenv import load_dotenv
from web3 import Web3

try:
    # web3 <7
    from web3.middleware import geth_poa_middleware

    def _inject_poa(w3: Web3):
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
except ImportError:
    # web3 >=7
    from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware

    def _inject_poa(w3: Web3):
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

load_dotenv()

env_rpc = os.getenv("RPC_URL") or os.getenv("ALCHEMY_RPC_URL")
env_chain = os.getenv("CHAIN_ID")
env_pk = os.getenv("PRIVATE_KEY")
env_addr = os.getenv("POLYGON_ADDRESS")

RPC_URL = env_rpc
CHAIN_ID = int(env_chain) if env_chain else 137
PRIVATE_KEY = env_pk

if not RPC_URL or not PRIVATE_KEY:
    try:
        from config import (
            RPC_URL as CFG_RPC,
            CHAIN_ID as CFG_CHAIN_ID,
            PRIVATE_KEY as CFG_PK,
            POLYGON_ADDRESS as CFG_ADDR,
            USDC_ADDRESS as CFG_USDC,
            CTF_CONTRACT_ADDRESS as CFG_CTF,
        )

        RPC_URL = RPC_URL or CFG_RPC
        CHAIN_ID = CHAIN_ID or CFG_CHAIN_ID
        PRIVATE_KEY = PRIVATE_KEY or CFG_PK
        env_addr = env_addr or CFG_ADDR
        USDC_ADDRESS = CFG_USDC
        CTF_CONTRACT_ADDRESS = CFG_CTF
    except Exception:
        pass

if not RPC_URL or not PRIVATE_KEY:
    raise SystemExit("RPC_URL and PRIVATE_KEY are required.")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
_inject_poa(w3)
account = w3.eth.account.from_key(PRIVATE_KEY)

if "USDC_ADDRESS" not in globals() or "CTF_CONTRACT_ADDRESS" not in globals():
    from config import USDC_ADDRESS, CTF_CONTRACT_ADDRESS

USDC_ADDRESS = Web3.to_checksum_address(USDC_ADDRESS)
CTF_CONTRACT_ADDRESS = Web3.to_checksum_address(CTF_CONTRACT_ADDRESS)

USDC_ABI = json.loads(
    '[{"constant":false,"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"}]'
)

usdc = w3.eth.contract(address=USDC_ADDRESS, abi=USDC_ABI)


def main():
    nonce = w3.eth.get_transaction_count(account.address)
    gas_price = w3.eth.gas_price
    tx = usdc.functions.approve(CTF_CONTRACT_ADDRESS, 2**256 - 1).build_transaction(
        {
            "from": account.address,
            "nonce": nonce,
            "gasPrice": gas_price,
            "chainId": CHAIN_ID,
        }
    )

    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    raw = getattr(signed, "rawTransaction", None) or signed.raw_transaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    print(f"✅ Approval submitted to CTF contract. Tx: {tx_hash.hex()}")
    print(f"   Owner:   {account.address}")
    print(f"   Spender: {CTF_CONTRACT_ADDRESS}")
    print(f"   Token:   {USDC_ADDRESS}")


if __name__ == "__main__":
    main()

