# Deployment Guide: Polymarket Gabagool Bot on AWS

This guide details how to deploy the bot to an AWS EC2 instance.

## Prerequisites
- An AWS Account
- SSH Key Pair (pem file) downloaded from AWS
- Your Polymarket API Credentials (`PRIVATE_KEY`, `POLYGON_ADDRESS`, `CLOB_API_KEY`, etc.)

## Step 1: Launch EC2 Instance

1.  **Go to AWS Console** -> **EC2** -> **Launch Instance**.
2.  **Name**: `polymarket-bot`
3.  **OS Image**: **Ubuntu Server 24.04 LTS** (Recommended).
4.  **Instance Type**: `t3.micro` (Free tier eligible) or `t3.small` (Better performance).
5.  **Key Pair**: Select your existing key pair or create a new one.
6.  **Network Settings**:
    *   Allow SSH traffic from "My IP".
7.  **Launch Instance**.

## Step 2: Prepare Local Files

Ensure you have your `.env` file ready with all credentials.

## Step 3: Deploy Code

Run the following commands from your local terminal (replace `your-key.pem` and `your-instance-ip`):

```bash
# 1. Fix permissions for your key (if needed)
chmod 400 path/to/your-key.pem

# 2. Copy files to the server
scp -i /Users/dardanberisha/polymarket-gabagool-bot/poly-bot.pem -r . ubuntu@98.86.148.148:~/POLYbot
```

*Note: You can exclude `venv` and `__pycache__` to speed up upload.*

## Step 4: Setup Server

SSH into the server:

```bash
ssh -i /Users/dardanberisha/polymarket-gabagool-bot/poly-bot.pem ubuntu@98.86.148.148
```

Run the setup commands (copy-paste block):

```bash
cd ~/POLYbot

# Update system
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv screen

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python3 check_now.py
```

## Step 5: Run the Bot

Use `screen` to keep the bot running even after you disconnect.

```bash
# Start a new screen session
screen -S bot

# Activate venv (if not active)
source venv/bin/activate

# Run the bot (runs indefinitely)
python3 main_hopper.py

# To detach (keep running in background): Press Ctrl+A, then D
```

To resume looking at the logs later:
```bash
screen -r bot
```

## Step 6: Troubleshooting

*   **Bot finds no markets?**
    *   Check if the instance time is synced: `date -u`
    *   Verify your `.env` has the correct `POLYGON_ADDRESS` and `PRIVATE_KEY`.
    *   Check if your IP is geo-blocked (run `curl ipinfo.io` on the server). AWS US East (N. Virginia) is usually fine.

*   **Bot crashes?**
    *   Check logs in `data/hopper_logs/`.

