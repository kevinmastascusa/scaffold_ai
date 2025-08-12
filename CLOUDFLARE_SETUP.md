# Cloudflare Tunnel Setup for ScaffoldAI

## Quick Start (One-Click Setup)

1. **Download the files:**
   - `ScaffoldAI-EnhancedUI.exe` (from GitHub release)
   - `start_cloudflare_tunnel.bat` (this repository)
   - Create `.env` file with your Hugging Face token

2. **Put all files in the same folder**

3. **Double-click `start_cloudflare_tunnel.bat`**

4. **Wait for the public URL to appear**

## Manual Setup

### Step 1: Install Cloudflare Tunnel
```bash
winget install Cloudflare.Cloudflared
```

### Step 2: Start Your App
```bash
# Run your ScaffoldAI executable
ScaffoldAI-EnhancedUI.exe
```

### Step 3: Start the Tunnel
```bash
# In a new terminal window
cloudflared tunnel --url http://localhost:5002
```

## What You'll See

```
2024-01-15T10:30:00Z INF Starting tunnel tunnelID=abc123
2024-01-15T10:30:00Z INF Version 2024.1.0
2024-01-15T10:30:00Z INF Requesting new quick tunnel on trycloudflare.com...
2024-01-15T10:30:01Z INF +----------------------------+
2024-01-15T10:30:01Z INF |  Your quick tunnel has been created!  |
2024-01-15T10:30:01Z INF |  URL: https://abc123.trycloudflare.com |
2024-01-15T10:30:01Z INF +----------------------------+
```

## Benefits

✅ **Completely Free** - No hosting costs  
✅ **Secure** - HTTPS encryption  
✅ **No Port Forwarding** - Works behind firewalls  
✅ **Global CDN** - Fast worldwide access  
✅ **No Configuration** - One command setup  

## Usage

1. **Share the URL** with professors/students
2. **Keep the terminal open** to maintain the tunnel
3. **Close the terminal** to stop the tunnel

## Troubleshooting

### App Won't Start
- Check that `.env` file exists with your Hugging Face token
- Ensure `ScaffoldAI-EnhancedUI.exe` is in the same folder
- Try running the EXE manually first

### Tunnel Won't Connect
- Make sure your app is running on `http://localhost:5002`
- Check your internet connection
- Try restarting the tunnel

### URL Not Working
- The tunnel URL changes each time you restart
- Make sure the terminal is still running
- Check that the app is still running locally

## Security Notes

- Your Hugging Face token stays on your local machine
- The tunnel is temporary and changes on restart
- Only share the URL with trusted users
- Close the tunnel when not in use

## Alternative: Permanent Tunnel

For a permanent URL, you can:
1. Sign up for free Cloudflare account
2. Create a named tunnel
3. Get a permanent subdomain

But the quick tunnel is perfect for demos and temporary sharing!
