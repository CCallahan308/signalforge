# Deploy SignalForge Streamlit Dashboard to Streamlit Cloud

## Overview

This guide shows how to deploy your SignalForge Streamlit dashboard to Streamlit Cloud for a live demo.

## Prerequisites

- GitHub account
- SignalForge repo: https://github.com/CCallahan308/signalforge
- Streamlit Cloud account (free)

## Step 1: Prepare Your Repository

Your repo already has:
- ✅ `src/app/dashboard.py` - Streamlit dashboard
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration

## Step 2: Create Streamlit Cloud Account

1. Go to: https://share.streamlit.io/
2. Sign in with GitHub
3. Authorize Streamlit Cloud

## Step 3: Deploy Your App

### Option A: Deploy from Streamlit Cloud UI

1. **Click "New app"** in Streamlit Cloud
2. **Configure:**
   - **Repository:** CCallahan308/signalforge
   - **Branch:** main
   - **Main file path:** `src/app/dashboard.py`
   - **Python version:** 3.11
3. **Click "Deploy"**
4. **Wait 2-3 minutes** for deployment
5. **Your app will be live at:** `https://signalforge-[random-id].streamlit.app/`

### Option B: Deploy from GitHub

1. **Add Streamlit Cloud badge to your README:**
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/signalforge/main/src/app/dashboard.py)
```

2. **Click the badge** to deploy
3. **Authorize** Streamlit Cloud
4. **Wait** for deployment

## Step 4: Configure Your App

### Update `.streamlit/config.toml` for cloud:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
headless = true
runOnSave = false

[browser]
gatherUsageStats = false
```

### Create `.streamlit/secrets.toml` (for sensitive data):

```toml
# Add database credentials if needed
# db_host = "your-host"
# db_user = "your-user"
# db_password = "your-password"
```

## Step 5: Update Portfolio Website

Update the Live Demo link in your portfolio to point to your Streamlit app:

**File:** `C:\Users\Calla\portfolio-website\app\projects\signalforge\page.tsx`

**Update the href:**
```tsx
href="https://signalforge-[your-id].streamlit.app/"
```

## Step 6: Commit and Push

```bash
cd C:\Users\Calla\signalforge
git add .
git commit -m "Add Streamlit Cloud deployment config"
git push

cd C:\Users\Calla\portfolio-website
git add .
git commit -m "Update SignalForge demo link to Streamlit Cloud"
git push
```

## Your Live Demo URL

After deployment, your app will be available at:
```
https://signalforge-[random-id].streamlit.app/
```

**Example:** `https://signalforge-churn-prediction.streamlit.app/`

## Benefits of Live Demo

✅ **Recruiters can try it** - Interactive, not just screenshots
✅ **Shows production readiness** - Deployed, not just local
✅ **Demonstrates functionality** - Working system, not just code
✅ **Easy to share** - One link to everything

## Troubleshooting

### App Won't Start

**Check:**
1. All dependencies in `requirements.txt`
2. Python version matches (3.11)
3. File path is correct (`src/app/dashboard.py`)
4. No local file paths in code

### Data Loading Errors

**Solution:** Use sample data or mock data for demo:
```python
# Use sample data instead of database connection
@st.cache_data
def load_sample_data():
    return pd.read_csv('data/processed/sample_data.csv')
```

### Memory Issues

**Solution:** Optimize data loading:
```python
# Load only necessary data
df = pd.read_csv('data/processed/features.csv', nrows=1000)
```

## Custom Domain (Optional)

To use a custom domain:
1. **In Streamlit Cloud:** Settings → Custom domain
2. **Add CNAME record:**
   ```
   signalforge.yourdomain.com → share.streamlit.io
   ```

## Monitoring Your App

**Streamlit Cloud provides:**
- ✅ Usage analytics
- ✅ Error logs
- ✅ Performance metrics
- ✅ Uptime monitoring

## Cost

**Free Tier:**
- Unlimited public apps
- Limited compute resources
- No custom domains

**Pro Tier ($250/year):**
- Private apps
- Custom domains
- More compute resources
- Priority support

## Next Steps

1. ✅ Deploy to Streamlit Cloud
2. ✅ Update portfolio website link
3. ✅ Share on LinkedIn
4. ✅ Add to resume

## Alternative: Other Hosting Options

If Streamlit Cloud doesn't work:

### **Heroku**
```bash
# Create Procfile
echo "web: streamlit run src/app/dashboard.py --server.port $PORT" > Procfile

# Deploy
heroku create signalforge-dashboard
git push heroku main
```

### **Railway**
```bash
railway init
railway run streamlit run src/app/dashboard.py
railway up
```

### **Render**
1. Connect GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run src/app/dashboard.py`

---

**Your Live Demo will be at:** `https://signalforge-[id].streamlit.app/`
