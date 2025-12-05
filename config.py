import os
from dotenv import load_dotenv  # type: ignore

load_dotenv()

# --- AUTHENTICATION ---
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
POLYGON_ADDRESS = os.getenv("POLYGON_ADDRESS")
API_KEY = os.getenv("CLOB_API_KEY")
API_SECRET = os.getenv("CLOB_API_SECRET")
API_PASSPHRASE = os.getenv("CLOB_API_PASSPHRASE")

# --- ENDPOINTS ---
REST_URL = "https://clob.polymarket.com"
WS_URL = "wss://ws-live-data.polymarket.com"

# --- POLYGON CHAIN ---
CHAIN_ID = 137
RPC_URL = "https://polygon-mainnet.g.alchemy.com/v2/sz8XTC13sVocHnsmQq9_R"

# --- STRATEGY SETTINGS ---
MAX_PAIR_COST = float(os.getenv("MAX_PAIR_COST", "1.0"))
ORDER_SIZE = float(os.getenv("ORDER_SIZE", "16"))  # gabagool22 uses 16 shares per order
STRATEGY_TYPE = os.getenv("STRATEGY_TYPE", "BUNDLE").upper()  # "BUNDLE" (recommended), "MOMENTUM", or "MAKER"

# =============================================================================
# BUNDLE ARBITRAGE SETTINGS (gabagool22's ACTUAL profitable strategy)
# Updated from 25,000 trade VWAP analysis - PNL_ANALYSIS_REPORT.md
# =============================================================================
# Core insight: Buy BOTH YES and NO for less than $1.00 = guaranteed profit
# PROFIT = (1.00 - bundle_cost) × min(yes_shares, no_shares)
#
# KEY FINDINGS FROM 25K TRADE ANALYSIS:
# - Avg Bundle VWAP: $0.987 (62.5% of markets < $1.00)
# - Avg 16.8 fills per transaction (multi-fill execution)
# - Price ranges: Up $0.09-$0.91, Down $0.11-$0.89
# - All 58 losses were UNHEDGED bets (100% loss rate on directional)
# - Realized ROI: +3.12% on $3,066 invested

# Bundle Cost Thresholds (UPDATED based on actual VWAP data)
BA_MAX_BUNDLE_COST = float(os.getenv("BA_MAX_BUNDLE_COST", "0.99"))      # Was 0.98, avg VWAP is 0.987
BA_TARGET_BUNDLE_COST = float(os.getenv("BA_TARGET_BUNDLE_COST", "0.985"))  # Was 0.97, realistic target
BA_ABORT_BUNDLE_COST = float(os.getenv("BA_ABORT_BUNDLE_COST", "0.995"))   # Tighter - losses happen >1.00

# Position Balance (winners avg 7.6% imbalance, losers 28.3%)
BA_MAX_IMBALANCE = float(os.getenv("BA_MAX_IMBALANCE", "0.10"))          # Max 10% difference YES vs NO
BA_MIN_POSITION_SIZE = float(os.getenv("BA_MIN_POSITION_SIZE", "100"))   # Don't bother with tiny positions
BA_MAX_TOTAL_EXPOSURE = float(os.getenv("BA_MAX_TOTAL_EXPOSURE", "2000")) # Max capital at risk

# Entry Price Thresholds (WIDENED based on actual price ranges $0.09-$0.91)
BA_MAX_YES_PRICE = float(os.getenv("BA_MAX_YES_PRICE", "0.85"))          # Was 0.75, widened (he buys up to $0.91)
BA_MAX_NO_PRICE = float(os.getenv("BA_MAX_NO_PRICE", "0.45"))            # Was 0.35, widened (he buys up to $0.89)

# Execution (avg 16.8 fills per transaction - AGGRESSIVE multi-fill)
BA_SWEEP_BURST_SIZE = int(os.getenv("BA_SWEEP_BURST_SIZE", "20"))        # Orders per burst wave
BA_SWEEP_INTERVAL_MS = float(os.getenv("BA_SWEEP_INTERVAL_MS", "5"))     # Was 10ms, faster execution
BA_ORDER_SIZE = float(os.getenv("BA_ORDER_SIZE", "16"))                  # Confirmed 16 shares standard
BA_MIN_ORDER_VALUE = float(os.getenv("BA_MIN_ORDER_VALUE", "1.0"))       # Minimum $1 per order
BA_FILLS_PER_SWEEP = int(os.getenv("BA_FILLS_PER_SWEEP", "17"))          # Target ~16.8 fills (from VWAP analysis)

# Time-based scoring (early morning markets have better arbitrage)
BA_MORNING_BONUS_START = int(os.getenv("BA_MORNING_BONUS_START", "9"))   # UTC hour start
BA_MORNING_BONUS_END = int(os.getenv("BA_MORNING_BONUS_END", "11"))      # UTC hour end
BA_MORNING_BONUS_MULT = float(os.getenv("BA_MORNING_BONUS_MULT", "1.1")) # 10% score bonus

# Enable/disable bundle arbitrage
ENABLE_BUNDLE_ARBITRAGE = os.getenv("ENABLE_BUNDLE_ARBITRAGE", "true").lower() == "true"

# --- DIRECTIONAL MAKER (new defaults) ---
DM_LEVELS = int(os.getenv("DM_LEVELS", "5"))               # ladder levels
DM_SIZE = float(os.getenv("DM_SIZE", "25"))                # shares per level (medium risk, loosened)
DM_TICK = float(os.getenv("DM_TICK", "0.01"))              # spacing between levels
DM_MAX_EXPOSURE = float(os.getenv("DM_MAX_EXPOSURE", "200"))  # max open size per market
DM_MIN_BID = float(os.getenv("DM_MIN_BID", "0.01"))        # skip if best bid below this (loosened)
DM_MAX_PRICE = float(os.getenv("DM_MAX_PRICE", "0.98"))    # do not post above this price (loosened)
DM_CANCEL_ON_SWITCH = os.getenv("DM_CANCEL_ON_SWITCH", "true").lower() == "true"

# MAKER / FISHER SPECIFIC SETTINGS (legacy, kept for backwards compat)
MAX_BUNDLE_COST = float(os.getenv("MAX_BUNDLE_COST", "0.98"))  # Cap for Bid(YES) + Bid(NO)
MIN_SPREAD = float(os.getenv("MIN_SPREAD", "0.02"))            # Target profit margin
TICK_SIZE = float(os.getenv("TICK_SIZE", "0.01"))              # Min price increment (1 cent)
REQUOTE_THRESHOLD = float(os.getenv("REQUOTE_THRESHOLD", "0.005")) # Don't update for noise < 0.5 cents

# --- MOMENTUM TAKER STRATEGY (mimics gabagool22) ---
# Price limits - gabagool22 buys YES up to $0.80, NO up to $0.55
# From analysis: gabagool22's avg YES fill = $0.646, avg NO fill = $0.336
MOMENTUM_YES_MAX_PRICE = float(os.getenv("MOMENTUM_YES_MAX_PRICE", "0.80"))
MOMENTUM_NO_MAX_PRICE = float(os.getenv("MOMENTUM_NO_MAX_PRICE", "0.55"))
MOMENTUM_NO_HEDGE_PRICE = float(os.getenv("MOMENTUM_NO_HEDGE_PRICE", "0.40"))  # Cheap hedge: buy NO below this

# Trend detection
TREND_WINDOW_SECONDS = float(os.getenv("TREND_WINDOW_SECONDS", "60"))  # 1-minute trend window
MIN_TREND_CHANGE = float(os.getenv("MIN_TREND_CHANGE", "0.02"))  # Min price change for trend signal

# Order timing - gabagool22 trades in bursts
BURST_COOLDOWN_MS = int(os.getenv("BURST_COOLDOWN_MS", "500"))  # Min time between orders (ms)
MAX_ORDERS_PER_BURST = int(os.getenv("MAX_ORDERS_PER_BURST", "5"))  # Max orders in rapid succession

# --- MARKET HOPPING SETTINGS (gabagool22's strategy) ---
# Enable hopping mode vs single-market mode
ENABLE_MARKET_HOPPING = os.getenv("ENABLE_MARKET_HOPPING", "true").lower() == "true"

# Scoring weights for opportunity comparison (from gabagool22 analysis)
# These determine which market gets focus
HOPPING_SPREAD_WEIGHT = float(os.getenv("HOPPING_SPREAD_WEIGHT", "0.30"))      # Tighter spread = better
HOPPING_MOMENTUM_WEIGHT = float(os.getenv("HOPPING_MOMENTUM_WEIGHT", "0.40"))  # Stronger trend = better
HOPPING_LIQUIDITY_WEIGHT = float(os.getenv("HOPPING_LIQUIDITY_WEIGHT", "0.20"))  # More depth = better
HOPPING_FILL_WEIGHT = float(os.getenv("HOPPING_FILL_WEIGHT", "0.10"))          # Recent fills = better

# Switching behavior
HOPPING_HYSTERESIS = float(os.getenv("HOPPING_HYSTERESIS", "0.20"))  # 20% advantage required to switch
FOCUS_SWITCH_COOLDOWN_MS = int(os.getenv("FOCUS_SWITCH_COOLDOWN_MS", "2000"))  # Min ms between switches

# Burst trading (gabagool22 averages 4.5 trades per burst)
BURST_SIZE = int(os.getenv("BURST_SIZE", "5"))  # Trades per burst before re-evaluating
BURST_INTERVAL_MS = int(os.getenv("BURST_INTERVAL_MS", "500"))  # Time between orders in burst

def _get_float(*keys, default):
    for key in keys:
        if key and (val := os.getenv(key)) is not None:
            return float(val)
    return float(default)


# Momentum / adaptive guard knobs (legacy - overridden by above if set)
MOMENTUM_NO_MIN_PRICE = _get_float(
    "MOMENTUM_NO_MIN_PRICE",
    "MOMENTUM_BUY_LOW_THRESHOLD_NO",
    default="0.85",
)
# NOTE: MOMENTUM_YES_MAX_PRICE is already defined above with correct default (0.80)
MOMENTUM_PRICE_BUFFER = float(os.getenv("MOMENTUM_PRICE_BUFFER", "0.01"))
MOMENTUM_TIMEOUT_SECONDS = int(os.getenv("MOMENTUM_TIMEOUT_SECONDS", "90"))
MOMENTUM_MAX_POSITIONS = int(os.getenv("MOMENTUM_MAX_POSITIONS", "3"))
MOMENTUM_SPREAD_MAX = float(os.getenv("MOMENTUM_SPREAD_MAX", "0.01"))
MOMENTUM_SPREAD_DISABLE = float(os.getenv("MOMENTUM_SPREAD_DISABLE", "0.05"))
ENABLE_MOMENTUM = os.getenv("ENABLE_MOMENTUM", "true").lower() == "true"

COMBINED_GUARD_WINDOW = int(os.getenv("COMBINED_GUARD_WINDOW", "40"))
COMBINED_GUARD_STD_MULT = float(os.getenv("COMBINED_GUARD_STD_MULT", "2.0"))
FEE_BUFFER_EXTRA = float(os.getenv("FEE_BUFFER_EXTRA", "0.0"))

MAX_UNHEDGED_EXPOSURE = float(os.getenv("MAX_UNHEDGED_EXPOSURE", "500"))
LOSS_LEASH = float(os.getenv("LOSS_LEASH", "-50"))
TRADING_MODE = os.getenv("TRADING_MODE", "ACTIVE").upper()
TRADE_LOG_PATH = os.getenv("TRADE_LOG_PATH", "data/trade_log.json")
MIN_PROBABILITY_FLOOR = float(os.getenv("MIN_PROBABILITY_FLOOR", "0.02"))

# --- MARKET OVERRIDES & SLUG PREFIXES ---
MARKET_SLUG = os.getenv("MARKET_SLUG")
MARKET_HANDOFF_BUFFER = int(os.getenv("MARKET_HANDOFF_BUFFER", "60"))
ETH_SLUG_PREFIX = os.getenv("ETH_SLUG_PREFIX", "eth-updown-15m")

BTC_MARKET_SLUG = os.getenv("BTC_MARKET_SLUG")
BTC_MARKET_HANDOFF_BUFFER = int(os.getenv("BTC_MARKET_HANDOFF_BUFFER", "60"))
BTC_SLUG_PREFIX = os.getenv("BTC_SLUG_PREFIX", "btc-updown-15m")

# --- MERGE / REDEMPTION SETTINGS ---
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_DECIMALS = 6
CTF_CONTRACT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
PARENT_COLLECTION_ID = "0x0000000000000000000000000000000000000000000000000000000000000000"
MERGE_PARTITION = [1, 2]
MERGE_GAS_LIMIT = 450000
TOKEN_DECIMALS = int(os.getenv("TOKEN_DECIMALS", str(USDC_DECIMALS)))
ENABLE_ONCHAIN_MERGE = os.getenv("ENABLE_ONCHAIN_MERGE", "true").lower() == "true"
ENABLE_ONCHAIN_REDEEM = os.getenv("ENABLE_ONCHAIN_REDEEM", "true").lower() == "true"
