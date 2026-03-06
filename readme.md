python -m streamlit run app.py

# Playwright (X/Twitter login fallback) setup:
# 1) pip install -r requirements.txt
# 2) python -m playwright install chromium
# 3) Optional env vars for authenticated reading:
#    X_USERNAME=your_x_username
#    X_PASSWORD=your_x_password
#    X_PLAYWRIGHT_STORAGE_STATE=x_storage_state.json
#    X_PLAYWRIGHT_HEADLESS=1
#    X_PLAYWRIGHT_TIMEOUT_MS=35000
