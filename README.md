# FDXPATTERN
Project of API from Angle one SmartAPI to access data and us for pattern prediction
# 📈 Triangle Pattern Detection using Angel One SmartAPI

This Python project detects **triangle chart patterns** (Symmetrical, Ascending, Descending) in stock price data fetched from Angel One's SmartAPI using hourly candles.

It processes a broad range of stocks across various sectors, identifies pivot points (swings), and runs linear regression to spot converging trendlines.

---

## 🚀 Features

- ✅ Auto-login to Angel One SmartAPI using TOTP-based 2FA
- 📦 Fetch historical 1-hour candle data
- 📍 Identify **pivot highs** and **pivot lows**
- 🔺 Detect:
  - Symmetrical Triangle
  - Ascending Triangle
  - Descending Triangle
- 🧠 Applies linear regression to validate converging trendlines
- 🧾 Clean output with symbol, pattern type, timestamp, close price, and regression stats

---

## 🛠 Requirements

- Python 3.7+
- `SmartApi` package from Angel One
- Other libraries:
  ```bash
  pip install requests pyotp pandas numpy scipy

 ## 📌 Notes
The script automatically filters out stocks with insufficient data or flat price lines.

You can tweak:

left_bars / right_bars for pivot sensitivity

min_r_value to control trendline fit strictness

backcandles window for how far to look for patterns

## Separate files for different patterns
🧠 Patterns Currently Supported
Pattern	Status	Notes

Triangle (Sym/Asc/Desc)	✅ Implemented	Uses regression on pivot points

Flag / Pennant	⏳ Planned	Rapid move + consolidation detection

Head & Shoulders	⏳ Planned	Detect neckline and 3-peak structure

Wedge Patterns	⏳ Planned	Converging lines in trend direction

Double Top / Bottom	⏳ Planned	M-pattern and W-pattern structure

## 4. Customize Watchlist (Optional)
Edit watchlist.py with your desired sector or stock groupings.

```
---
=== Processing: RELIANCE ===
✅ Data processed successfully. Found 43 pivot points.
🔺 Found 1 triangle pattern for RELIANCE!
RELIANCE | Symmetrical Triangle | CandleID: 65 | Time: 2025-06-14 13:00 | Close: ₹2873.90 | HighR: 0.911 | LowR: 0.844

```
