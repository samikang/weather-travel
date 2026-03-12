from datetime import date as _date
from pathlib import Path
from models import TripContext

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ heavy hail",
}


def _fmt_date(d: str) -> str:
    try:
        return _date.fromisoformat(d).strftime("%a %d %b")
    except Exception:
        return d


def _method_label(method: str, n_years: int = None) -> str:
    if method == "historical":
        return f"Historical Prediction ({n_years}yr avg + trend)"
    if method == "current":
        return "Current-Conditions Prediction (anomaly + decay)"
    return "Live Forecast"


# ── Matplotlib forecast chart ─────────────────────────────────────────────────

def plot_forecast(forecasts: list, context: TripContext,
                  start_date: str, end_date: str,
                  method: str = "forecast", n_years: int = None) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    dates    = [_date.fromisoformat(f.date) for f in forecasts]
    temp_min = np.array([f.temp_min for f in forecasts])
    temp_max = np.array([f.temp_max for f in forecasts])
    temp_avg = (temp_min + temp_max) / 2
    precip   = np.array([f.precipitation_mm for f in forecasts])
    uv       = np.array([f.uv_index_max for f in forecasts])

    title = (f"{context.city}, {context.country}  |  "
             f"{start_date} → {end_date}  |  {_method_label(method, n_years)}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # Panel 1: Temperature
    ax1 = axes[0]
    ax1.fill_between(dates, temp_min, temp_max, alpha=0.25, color="tomato", label="Min–Max range")
    ax1.plot(dates, temp_avg, color="crimson",   linewidth=2, marker="o", markersize=4, label="Avg")
    ax1.plot(dates, temp_min, color="steelblue", linewidth=1, linestyle="--", alpha=0.6, label="Min")
    ax1.plot(dates, temp_max, color="darkorange",linewidth=1, linestyle="--", alpha=0.6, label="Max")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Precipitation
    ax2 = axes[1]
    colors = ["#d32f2f" if p > 20 else "#1976d2" if p > 1 else "#90caf9" for p in precip]
    ax2.bar(dates, precip, color=colors, alpha=0.8, width=0.6, label="Precipitation")
    ax2.axhline(1,  color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="1mm")
    ax2.axhline(20, color="red",  linewidth=0.8, linestyle="--", alpha=0.5, label="Heavy (20mm)")
    ax2.set_ylabel("Precipitation (mm)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: UV Index
    ax3 = axes[2]
    ax3.axhspan(0,  3,  alpha=0.07, color="green")
    ax3.axhspan(3,  6,  alpha=0.07, color="yellow")
    ax3.axhspan(6,  8,  alpha=0.07, color="orange")
    ax3.axhspan(8,  16, alpha=0.07, color="red")
    for level, lbl, color in [(3, "Moderate (3)", "gold"), (6, "High (6)", "orange"), (8, "Very High (8)", "red")]:
        ax3.axhline(level, color=color, linewidth=0.8, linestyle="--", alpha=0.6, label=lbl)
    ax3.plot(dates, uv, color="darkorange", linewidth=2, marker="s", markersize=4, label="UV Index")
    ax3.set_ylabel("UV Index")
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    city_slug = context.city.lower().replace(" ", "_").replace(",", "")
    out_path = Path(__file__).parent / f"{city_slug}_{start_date}_forecast.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ── Terminal display ──────────────────────────────────────────────────────────

def display_rich(forecasts: list, context: TripContext,
                 start_date: str, end_date: str,
                 method: str = "forecast", n_years: int = None):
    console = Console()

    if method == "historical":
        method_tag = f"[yellow]{_method_label(method, n_years)}[/]"
    elif method == "current":
        method_tag = f"[cyan]{_method_label(method)}[/]"
    else:
        method_tag = f"[green]{_method_label(method)}[/]"

    header = (
        f"[bold]Trip:[/] {context.city}, {context.country}  |  "
        f"[bold]Dates:[/] {_fmt_date(start_date)} → {_fmt_date(end_date)}  |  "
        f"[bold]Method:[/] {method_tag}"
    )
    console.print(Panel(header, title="[bold cyan]Travel Weather Advisor[/]", border_style="cyan"))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Date",        style="cyan",  no_wrap=True)
    table.add_column("Condition",   style="white")
    table.add_column("Temp (°C)",   style="white", no_wrap=True)
    table.add_column("Rain (mm)",   style="white", no_wrap=True)
    table.add_column("UV",          style="white", no_wrap=True)
    table.add_column("Wind (km/h)", style="white", no_wrap=True)
    table.add_column("Cloud (%)",   style="white", no_wrap=True)

    for f in forecasts:
        condition = WMO_CODES.get(f.weather_code, f"Code {f.weather_code}")
        temp_str  = f"{f.temp_min:.0f} – {f.temp_max:.0f}"
        rain_str  = f"[red]{f.precipitation_mm:.1f}[/]" if f.precipitation_mm > 20 \
                    else f"[yellow]{f.precipitation_mm:.1f}[/]" if f.precipitation_mm > 1 \
                    else f"{f.precipitation_mm:.1f}"
        uv_str    = f"[red]{f.uv_index_max:.0f}[/]"    if f.uv_index_max > 6 \
                    else f"[yellow]{f.uv_index_max:.0f}[/]" if f.uv_index_max > 3 \
                    else f"{f.uv_index_max:.0f}"
        wind_str  = f"[red]{f.wind_speed_max:.0f}[/]"  if f.wind_speed_max > 60 \
                    else f"[yellow]{f.wind_speed_max:.0f}[/]" if f.wind_speed_max > 40 \
                    else f"{f.wind_speed_max:.0f}"
        cloud_str = f"{f.cloud_cover_mean:.0f}"

        table.add_row(_fmt_date(f.date), condition, temp_str,
                      rain_str, uv_str, wind_str, cloud_str)

    console.print(table)


def display_plain(forecasts: list, context: TripContext,
                  start_date: str, end_date: str,
                  method: str = "forecast", n_years: int = None):
    sep = "-" * 70
    print(sep)
    print("Travel Weather Advisor")
    print(f"Trip  : {context.city}, {context.country}")
    print(f"Dates : {start_date} to {end_date}")
    print(f"Method: {_method_label(method, n_years)}")
    print(sep)
    print(f"{'Date':<14} {'Condition':<22} {'Temp':>10} {'Rain':>8} {'UV':>4} {'Wind':>6} {'Cloud':>6}")
    print(sep)
    for f in forecasts:
        condition = WMO_CODES.get(f.weather_code, f"Code {f.weather_code}")
        print(f"{_fmt_date(f.date):<14} {condition:<22} "
              f"{f.temp_min:.0f}–{f.temp_max:.0f}°C{'':{'>4'}} "
              f"{f.precipitation_mm:>6.1f}mm "
              f"{f.uv_index_max:>4.0f} "
              f"{f.wind_speed_max:>5.0f} "
              f"{f.cloud_cover_mean:>5.0f}%")
    print(sep)


def display(forecasts: list, context: TripContext,
            start_date: str, end_date: str,
            method: str = "forecast", n_years: int = None):
    if RICH_AVAILABLE:
        display_rich(forecasts, context, start_date, end_date, method, n_years)
    else:
        display_plain(forecasts, context, start_date, end_date, method, n_years)
