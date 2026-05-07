# DeepVerify Pro Max

FastAPI + React/Vite tabanlı, ChatGPT benzeri arayüze sahip kanıt odaklı dezenformasyon doğrulama sistemi.

## Pro katmanlar
- Orijinal `agent_logic.py` ve `utils.py` korunmuştur.
- SerpAPI + Gemini RAG akışı ana algoritmadır.
- Opsiyonel NewsAPI canlı haber sinyali eklendi.
- X/Twitter için SerpAPI public-index araması eklendi.
- Fake-news pattern dataset destekli risk skoru eklendi.
- Kaynak kalite ağırlığı ve güven skoru yeniden kalibrasyonu eklendi.
- Login/register, SQLite, sohbet geçmişi ve ChatGPT tarzı UI vardır.

## Çalıştırma

Backend:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

## .env
`backend/.env.example` dosyasını `backend/.env` olarak kopyala ve keyleri yaz.


## NewsAPI Key Nasıl Alınır?
1. https://newsapi.org/register adresine gir.
2. Ücretsiz hesap oluştur ve e-posta doğrulamasını yap.
3. Dashboard ekranındaki API key'i kopyala.
4. `backend/.env` içine şunu ekle:

```env
NEWS_API_KEY=BURAYA_NEWSAPI_KEY
```

NewsAPI opsiyoneldir. Key yoksa proje SerpAPI + Gemini ile çalışmaya devam eder; key eklenirse canlı haber sinyali de algoritmaya dahil edilir.

## v8 Güncellemesi
- Detaylı analiz varsayılan olarak kapalı gelir; tıklanınca açılır.
- UI bozulmadan ChatGPT tarzı sade sonuç akışı korunur.
- `.env.example` içinde NewsAPI ve detaylı analiz ayarları hazırdır.
