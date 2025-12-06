from utils import get_target_markets
print("Checking for active markets...")
btc = get_target_markets(asset_label="BTC", slug_prefix="btc-updown-15m")
print(f"BTC: {btc}")

