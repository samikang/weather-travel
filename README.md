# Travel Weather Advisor

A command-line tool that forecasts weather for any city and date range using three different prediction methods — live API, current-conditions prediction, or long-range historical climate prediction.

---

## Features

- Look up any city worldwide by name (automatic geocoding)
- Three forecast methods: live API, current-conditions, or historical climate
- Per-day weather table: condition, temperature, precipitation, UV, wind, cloud cover
- Colour-coded terminal output (red/yellow highlights for high-risk values)
- **Matplotlib** 3-panel forecast chart saved as PNG
- **Parquet cache** — historical data cached on disk, repeated queries load instantly

---

## Requirements

- Python 3.9+
- Internet connection (for weather data)

Install all dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---|---|
| `rich` | Colour terminal output (tables, panels) |
| `numpy` | Weighted mean, OLS trend, array operations |
| `pandas` | DataFrame storage for multi-year historical observations |
| `matplotlib` | 3-panel forecast chart (temperature, precipitation, UV) |
| `pyarrow` | Parquet engine — cache historical API responses on disk |

No API keys required. All weather data is sourced from [Open-Meteo](https://open-meteo.com/), which is free and open-access.

---

## Usage

```bash
python main.py --city CITY --start YYYY-MM-DD --end YYYY-MM-DD [--method METHOD] [--years N] [--chart]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--city` | Yes | Destination city, e.g. `"Tokyo"` or `"Paris, France"` |
| `--start` | Yes | Travel start date in `YYYY-MM-DD` format |
| `--end` | Yes | Travel end date in `YYYY-MM-DD` format |
| `--method` | No | `forecast` (default), `current`, or `historical` |
| `--years` | No | Years of archive data to use (default: `10`, historical mode only) |
| `--chart` | No | Save a 3-panel matplotlib forecast chart as PNG |

### Examples

```bash
# Live forecast
python main.py --city "Singapore" --start 2026-03-15 --end 2026-03-20
python main.py --city "Tokyo"     --start 2026-03-15 --end 2026-03-20 --method forecast

python main.py --city "Tokyo"  --start 2026-07-10 --end 2026-07-17 --method historical
```

---

## Forecast Methods

### `forecast` (default)

Retrieves real meteorological model output directly from the **Open-Meteo forecast API**. This is the most accurate option for near-future trips.

- Maximum range: **16 days ahead**
- No custom algorithm — returns the live NWP (Numerical Weather Prediction) model output

---

### `historical`

A **custom prediction algorithm** that estimates weather based solely on the same calendar period from past years. Use this for trips further than 16 days ahead.

Historical API responses are cached as **Parquet files** (`cache/`) — re-running the same city and date range loads from disk instantly.

**Algorithm — Exponentially Weighted Climatology + OLS Trend:**

---

## Output Table

Each method produces the same weather table:

| Column | Description |
|---|---|
| Date | Day of the trip |
| Condition | WMO weather description (e.g. Slight rain, Overcast) |
| Temp (°C) | Min – Max temperature range |
| Rain (mm) | Daily precipitation; yellow > 1mm, red > 20mm |
| UV | UV index max; yellow > 3, red > 6 |
| Wind (km/h) | Max wind speed; yellow > 40, red > 60 |
| Cloud (%) | Mean cloud cover |

---

## Forecast Chart (`--chart`)

Saves a 3-panel PNG to the project directory named `{city}_{start_date}_forecast.png`.

| Panel | Content |
|---|---|
| Temperature | Min/max shaded band + average, min, max lines |
| Precipitation | Bar chart; blue = light rain, red = heavy rain (> 20mm) |
| UV Index | Line chart with green/yellow/orange/red risk-level shading |

---

## Data Sources

| Source | Usage | Cost |
|---|---|---|
| [Open-Meteo Forecast API](https://api.open-meteo.com) | Live 16-day weather forecast | Free, no key |
| [Open-Meteo Archive API](https://archive-api.open-meteo.com) | Historical weather data (1940–present) | Free, no key |
| [Open-Meteo Geocoding API](https://geocoding-api.open-meteo.com) | City name to coordinates | Free, no key |

---

## Limitations

- `forecast` is limited to **16 days ahead**
- `current` requires the archive API to have data up to ~2 days ago; very recent data may occasionally be unavailable
- `historical` accuracy decreases for unusual weather years — outliers are partially smoothed by the weighted mean
- UV data in the historical archive is sparse before ~2019; a latitude/cloud proxy is used as fallback
