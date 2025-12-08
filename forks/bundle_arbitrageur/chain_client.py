import asyncio
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import List, Optional

from web3 import Web3

from config import (
    PRIVATE_KEY,
    POLYGON_ADDRESS,
    RPC_URL,
    CHAIN_ID,
    USDC_ADDRESS,
    CTF_CONTRACT_ADDRESS,
    PARENT_COLLECTION_ID,
    MERGE_PARTITION,
    MERGE_GAS_LIMIT,
    TOKEN_DECIMALS,
    ENABLE_ONCHAIN_MERGE,
    ENABLE_ONCHAIN_REDEEM,
)


# Minimal ABI for the Conditional Tokens Framework merge/redeem functions.
CTF_ABI = json.loads(
    """
    [
        {
            "name": "mergePositions",
            "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "partition", "type": "uint256[]"},
                {"name": "amount", "type": "uint256"}
            ],
            "outputs": []
        },
        {
            "name": "redeemPositions",
            "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"}
            ],
            "outputs": []
        }
    ]
    """
)


@dataclass
class OnchainResult:
    action: str
    tx_hash: Optional[str]
    error: Optional[str] = None


def _to_bytes32(value: str) -> bytes:
    return Web3.to_bytes(hexstr=value)


def _shares_to_units(shares: float) -> int:
    """
    Convert human-readable share count to on-chain units.
    Polymarket conditional tokens use the same decimal precision as collateral (USDC).
    Default is 6 decimals but override via TOKEN_DECIMALS if needed.
    """
    quant = Decimal(shares)
    factor = Decimal(10) ** Decimal(TOKEN_DECIMALS)
    return int((quant * factor).to_integral_value(rounding=ROUND_DOWN))


class CTFSettlementClient:
    """
    Thin web3 helper to call mergePositions and redeemPositions on the
    Polymarket Conditional Tokens Framework. All calls are gated by
    ENABLE_ONCHAIN_MERGE / ENABLE_ONCHAIN_REDEEM so users can opt in safely.
    """

    def __init__(self):
        self.enabled_merge = ENABLE_ONCHAIN_MERGE
        self.enabled_redeem = ENABLE_ONCHAIN_REDEEM
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        self.account = self.w3.eth.account.from_key(PRIVATE_KEY)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(CTF_CONTRACT_ADDRESS),
            abi=CTF_ABI,
        )

    async def merge_positions(self, condition_id: str, shares: float) -> OnchainResult:
        if not self.enabled_merge:
            return OnchainResult(action="merge", tx_hash=None, error="onchain merge disabled")
        if shares <= 0:
            return OnchainResult(action="merge", tx_hash=None, error="no hedged shares to merge")

        amount = _shares_to_units(shares)
        parent = _to_bytes32(PARENT_COLLECTION_ID)
        condition = _to_bytes32(condition_id)
        partition = [int(x) for x in MERGE_PARTITION]

        def _send():
            tx = self.contract.functions.mergePositions(
                Web3.to_checksum_address(USDC_ADDRESS),
                parent,
                condition,
                partition,
                amount,
            ).build_transaction(
                {
                    "from": self.account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                    "gas": MERGE_GAS_LIMIT,
                    "chainId": CHAIN_ID,
                }
            )
            signed = self.account.sign_transaction(tx)
            return self.w3.eth.send_raw_transaction(signed.rawTransaction).hex()

        try:
            tx_hash = await asyncio.to_thread(_send)
            print(f"[Onchain] 🔀 mergePositions sent (shares={shares:.4f}) tx={tx_hash}")
            return OnchainResult(action="merge", tx_hash=tx_hash)
        except Exception as exc:  # pylint: disable=broad-except
            err = str(exc)
            print(f"[Onchain] mergePositions error: {err}")
            return OnchainResult(action="merge", tx_hash=None, error=err)

    async def redeem_positions(
        self, condition_id: str, index_sets: List[int]
    ) -> OnchainResult:
        if not self.enabled_redeem:
            return OnchainResult(action="redeem", tx_hash=None, error="onchain redeem disabled")
        if not index_sets:
            return OnchainResult(action="redeem", tx_hash=None, error="no index sets provided")

        parent = _to_bytes32(PARENT_COLLECTION_ID)
        condition = _to_bytes32(condition_id)
        index_sets_int = [int(x) for x in index_sets]

        def _send():
            tx = self.contract.functions.redeemPositions(
                Web3.to_checksum_address(USDC_ADDRESS),
                parent,
                condition,
                index_sets_int,
            ).build_transaction(
                {
                    "from": self.account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                    "chainId": CHAIN_ID,
                }
            )
            signed = self.account.sign_transaction(tx)
            return self.w3.eth.send_raw_transaction(signed.rawTransaction).hex()

        try:
            tx_hash = await asyncio.to_thread(_send)
            joined = ",".join(str(i) for i in index_sets_int)
            print(f"[Onchain] 🏁 redeemPositions sent (index_sets={joined}) tx={tx_hash}")
            return OnchainResult(action="redeem", tx_hash=tx_hash)
        except Exception as exc:  # pylint: disable=broad-except
            err = str(exc)
            print(f"[Onchain] redeemPositions error: {err}")
            return OnchainResult(action="redeem", tx_hash=None, error=err)

