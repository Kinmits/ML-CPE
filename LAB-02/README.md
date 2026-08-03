# ⚽ FPL Data Analysis & Preprocessing Project

โปรเจกต์นี้เป็นการสำรวจ ตรวจสอบคุณภาพข้อมูล (Data Quality Audit) และจัดเตรียมข้อมูล (Data Preprocessing) สำหรับชุดข้อมูลผู้เล่นฟุตบอล **Fantasy Premier League (FPL)** เพื่อนำไปใช้ในการวิเคราะห์เชิงลึกและการสร้างโมเดลพยากรณ์คะแนนผู้เล่น

---

## 📁 ข้อมูลชุดข้อมูล (Dataset Information)

- **ชื่อชุดข้อมูล:** `fpl.csv`
- **จำนวนแถวทั้งหมด (Records):** 567 คน [cite: 1]
- **จำนวนคุณลักษณะทั้งหมด (Features):** 77 คอลัมน์ [cite: 1]
- **ตัวแปรเป้าหมายหลัก:** `total_points`, `points_per_game`, `goals_scored`, `assists`, `now_cost`

---

## 📊 สรุปผลการสำรวจและเตรียมข้อมูล (Summary of Exploration & Preprocessing)

### 1. โครงสร้างและประเภทข้อมูล
- **ขนาดข้อมูล:** `(567, 77)` [cite: 1]
- **ประเภทข้อมูล:**
  - `int64`: 40 คอลัมน์ [cite: 1]
  - `float64`: 33 คอลัมน์ [cite: 1]
  - `str / object`: 4 คอลัมน์ (`player_name`, `club_name`, `position_name`, `news`) [cite: 1]

### 2. การกระจายตัวตามตำแหน่งผู้เล่น (Position Distribution)
| ตำแหน่ง (Position) | ชื่อเต็ม | จำนวน (คน) | ร้อยละ (%) |
| :---: | :--- | :---: | :---: |
| **MID** | กองกลาง (Midfielder) | 250 | 44.09% [cite: 1] |
| **DEF** | กองหลัง (Defender) | 186 | 32.80% [cite: 1] |
| **FWD** | กองหน้า (Forward) | 68 | 11.99% [cite: 1] |
| **GKP** | ผู้รักษาประตู (Goalkeeper) | 63 | 11.11% [cite: 1] |

### 3. การตรวจสอบข้อมูลที่ขาดหาย (Missing Values Summary)
พบข้อมูลขาดหายใน 7 คอลัมน์หลัก:
- `corners_and_indirect_freekicks_text`: 100% (567/567) [cite: 1]
- `penalties_text`: 100% (567/567) [cite: 1]
- `direct_freekicks_text`: 100% (567/567) [cite: 1]
- `news`: 89.77% (509/567) [cite: 1]
- `direct_freekicks_order`: 89.59% (508/567) [cite: 1]
- `penalties_order`: 88.71% (503/567) [cite: 1]
- `corners_and_indirect_freekicks_order`: 87.48% (496/567) [cite: 1]

*หมายเหตุ: ไม่พบข้อมูลซ้ำซ้อน (Duplicates = 0)* [cite: 1]

---

## 🏆 สรุปผู้เล่นทำคะแนนสูงสุด (Top Performers Summary)

### 5 อันดับแรกคะแนนรวมสูงสุด (Top 5 Overall Points)
1. **Erling Haaland** (Man City | FWD) – **239 คะแนน** (£15.5m) [cite: 1]
2. **Bruno B.Fernandes** (Man Utd | MID) – **235 คะแนน** (£12.0m) [cite: 1]
3. **Gabriel Gabriel** (Arsenal | DEF) – **209 คะแนน** (£8.0m) [cite: 1]
4. **Antoine Semenyo** (Man City | MID) – **202 คะแนน** (£8.5m) [cite: 1]
5. **Morgan Gibbs-White** (Nott'm Forest | MID) – **188 คะแนน** (£8.0m) [cite: 1]

---

## 🚀 วิธีการรันโปรเจกต์ (How to Run)

1. **โคลนหรือดาวน์โหลด Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **ติดตั้งไลบรารีที่จำเป็น:**
   ```bash
   pip install pandas numpy matplotlib seaborn weasyprint
   ```

3. **เปิดและรันไฟล์ Jupyter Notebook หรือสคริปต์:**
   ```bash
   jupyter notebook
   ```

---

## 🛠️ แผนการเตรียมข้อมูลก่อนทำโมเดล (Data Preprocessing Pipeline)

1. **Feature Dropping:** ลบคอลัมน์ที่มีค่าสูญหาย 100% (`corners_and_indirect_freekicks_text`, `penalties_text`, `direct_freekicks_text`)
2. **Missing Value Imputation:** แทนที่ค่า NaN ในคอลัมน์ลำดับการเตะลูกนิ่งด้วย `0` และคอลัมน์ข่าวด้วย `'No News'`
3. **Categorical Encoding:** แปลงตัวแปร `position_name` และ `club_name` เป็นรูปแบบตัวเลข (One-Hot Encoding / Label Encoding)
4. **Feature Scaling:** ปรับสเกลข้อมูลเชิงตัวเลขด้วย MinMaxScaler หรือ StandardScaler เพื่อเตรียมพร้อมสำหรับกระบวนการ Machine Learning
