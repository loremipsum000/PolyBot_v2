#!/usr/bin/env python3
"""Test script to verify market initialization."""

import asyncio
import sys
sys.path.insert(0, '.')

async def test():
    from market_hopper import MarketHopper
    
    hopper = MarketHopper()
    print('\n=== Testing Market Initialization ===')
    
    try:
        await hopper.initialize_markets()
        
        print('\n=== Market Summary ===')
        for label, market in hopper.markets.items():
            print(f'{label}:')
            print(f'  Slug: {market.slug}')
            print(f'  Condition ID: {market.condition_id[:20]}...' if market.condition_id else '  Condition ID: MISSING')
            print(f'  YES Token: {market.yes_token[:20]}...' if market.yes_token else '  YES Token: MISSING')
            print(f'  NO Token: {market.no_token[:20]}...' if market.no_token else '  NO Token: MISSING')
            print(f'  End: {market.end_date_iso}')
            
        # Now test order book fetch
        print('\n=== Testing Order Book Fetch ===')
        await hopper.poll_market_data()
        
        for label, market in hopper.markets.items():
            print(f'{label}: Ask YES=${market.best_ask_yes:.3f}, Ask NO=${market.best_ask_no:.3f}, Bundle=${market.bundle_cost:.3f}')
            print(f'       Has Liquidity: {market.has_liquidity}, Has Arbitrage: {market.has_arbitrage}')
            
    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        
    hopper.stop_event.set()
    if hopper.ws_task:
        hopper.ws_task.cancel()

if __name__ == '__main__':
    asyncio.run(test())
