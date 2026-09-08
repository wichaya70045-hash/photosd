# AI Photo Selection System — PHOTOSS

ระบบคัดเลือกภาพงานราชกิจฯ แบบกึ่งอัตโนมัติด้วย AI  
**ทำงาน 100% Offline บน Laptop (CPU เท่านั้น)**

---

## 🚀 เริ่มใช้งาน (Windows)

### วิธีที่ 1: Double-click
```
ดับเบิลคลิก start.bat
```

### วิธีที่ 2: Command Line
```bash
cd photoss
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

เปิดเบราว์เซอร์: **http://localhost:8000**

---

## 📂 การนำเข้ารูปภาพ (2 วิธี)

### วิธีที่ 1: ตรวจสอบโฟลเดอร์ในเครื่องโดยตรง (แนะนำ — รวดเร็วที่สุด ⚡)
1. นำไฟล์ภาพจากการ์ดกล้อง/Flash drive มาวางในโฟลเดอร์ในเครื่อง (เช่น `C:\Photos\...`)
2. เปิดเบราว์เซอร์ **http://localhost:8000**
3. เลือกแท็บ **"📂 สแกนโฟลเดอร์ในเครื่อง"**
4. ระบุ Folder Path หรือกดเลือกปุ่มทางลัด (Pictures, Downloads, Desktop)
5. กด **"🚀 เริ่มสแกน & ตรวจสอบภาพในเครื่องทันที"**
* ไม่ต้องเสียเวลารออัปโหลดผ่านเครือข่าย
* ระบบอ่านไฟล์จากดิสก์โดยตรงและประมวลผลทันที

### วิธีที่ 2: อัปโหลดผ่าน WiFi Hotspot (จากมือถือ/iPhone)
1. เปิด **Personal Hotspot** จาก Laptop ของคุณ
2. เชื่อมต่อมือถือเข้า Hotspot
3. หาค่า IP ของ Laptop: `ipconfig` → IPv4 ที่ Adapter hotspot
4. เปิดเบราว์เซอร์บนมือถือ → `http://<IP>:8000`
5. อัปโหลดภาพผ่านหน้าเว็บ (รองรับ JPG, PNG, HEIC)


---

## 🗂 โครงสร้างโปรเจค

```
photoss/
├── config.yaml          ← ปรับ threshold ที่นี่
├── requirements.txt
├── main.py              ← entry point
├── start.bat            ← เริ่มระบบ (Windows)
├── app/
│   ├── core/            ← config + database
│   ├── models/          ← ORM models
│   ├── services/        ← business logic
│   │   ├── quality_filter.py   (blur/exposure/obstruction)
│   │   ├── clustering.py       (EXIF time + perceptual hash)
│   │   ├── ml_scorer.py        (MobileNetV3 + classifier)
│   │   ├── ingestion.py        (watchdog + HEIC + pipeline)
│   │   └── retraining.py       (continuous learning)
│   ├── api/             ← FastAPI routers
│   └── static/          ← Frontend (HTML + CSS + JS)
├── data/
│   ├── photos.db        ← SQLite database
│   ├── models/          ← saved ML classifiers
│   ├── watch_inbox/     ← Watched folder (SD card / USB)
│   └── uploads/         ← Staging area
└── tests/
    └── test_quality_filter.py
```

---

## ⚙️ ปรับแต่ง Threshold (`config.yaml`)

| ค่า | ค่าเริ่มต้น | คำอธิบาย |
|-----|------------|-----------|
| `blur_threshold` | 100 | Laplacian variance; ต่ำกว่านี้ = ภาพเบลอ |
| `exposure_over_threshold` | 0.15 | สัดส่วน pixel สว่างเกิน (overexposed) |
| `exposure_under_threshold` | 0.15 | สัดส่วน pixel มืดเกิน (underexposed) |
| `obstruction_edge_density` | 0.05 | Edge density ต่ำสุดในกลางภาพ |
| `time_gap_minutes` | 15 | ช่วงห่างระหว่างภาพเพื่อแยกกิจกรรม |
| `cold_start_min_samples` | 50 | ต้องมีตัวอย่างอย่างน้อยเท่านี้จึงเปิด ML |
| `retrain_every_n_batches` | 5 | Retrain ทุก N batch ที่สำเร็จ |
| `top_n_per_batch` | 20 | จำนวนภาพ Shortlist สูงสุด |

---

## 🔄 สาม Phase การพัฒนา

| Phase | ฟีเจอร์ | สถานะ |
|-------|---------|-------|
| **Phase 1** | Ingestion + Technical Filter + Review UI | ✅ พร้อม |
| **Phase 2** | ML Scoring (MobileNetV3 + LogisticRegression) | ✅ พร้อม |
| **Phase 3** | Continuous Learning + Versioning | ✅ พร้อม |

---

## 🧪 รัน Unit Tests

```bash
cd photoss
venv\Scripts\activate
pip install pytest
pytest tests/ -v
```

---

## 📚 Tech Stack

- **Backend**: Python 3.11+, FastAPI, uvicorn
- **ML**: torchvision MobileNetV3-Small (frozen), scikit-learn
- **Image**: OpenCV, pillow-heif (HEIC support)
- **DB**: SQLite + SQLAlchemy
- **Frontend**: HTML + Alpine.js (no build tools)
- **Watch**: watchdog library

---

## 🔗 API Endpoints

| Method | Path | คำอธิบาย |
|--------|------|----------|
| `POST` | `/api/batches` | สร้าง Batch ใหม่ |
| `POST` | `/api/upload/{batch_id}` | อัปโหลดภาพ |
| `POST` | `/api/batches/{id}/complete` | ส่งประมวลผล + Cluster |
| `GET`  | `/api/batches/{id}/review` | ดูภาพจัดกลุ่ม + Shortlist |
| `POST` | `/api/batches/{id}/selections` | บันทึกการคัดเลือก |
| `GET`  | `/api/stats` | สถิติระบบ |
| `POST` | `/api/retrain` | Retrain ML ทันที |
| `GET`  | `/docs` | Swagger API docs |
