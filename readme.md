# Kurulum (Windows PowerShell)

1. Sanal ortam oluştur:
   `python -m venv venv`

2. Sanal ortamı aktif et:
   `venv\Scripts\Activate.ps1`

3. Bağımlılıkları kur:
   `pip install -r requirements.txt`

4. Playwright tarayıcısını kur:
   `python -m playwright install chromium`

5. Uygulamayı çalıştır:
   `python -m streamlit run app.py`

## Opsiyonel: X/Twitter girişli okuma için .env

- `X_USERNAME=your_x_username`
- `X_PASSWORD=your_x_password`
- `X_PLAYWRIGHT_STORAGE_STATE=x_storage_state.json`
- `X_PLAYWRIGHT_HEADLESS=1`
- `X_PLAYWRIGHT_TIMEOUT_MS=35000`
