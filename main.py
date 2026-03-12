#!/usr/bin/env python3
"""
Travel Weather Advisor
======================
Usage:
    python main.py --city "Singapore" --start 2026-03-15 --end 2026-03-20 --method forecast
    python main.py --city "Tokyo"     --start 2026-03-15 --end 2026-03-20 --method current
    python main.py --city "Tokyo"     --start 2026-07-10 --end 2026-07-17 --method historical
    python main.py --city "London"    --start 2026-12-20 --end 2026-12-25 --method historical --chart
"""

import argparse
import sys
from datetime import date, timedelta

from geocoder import get_location
from weather import get_forecast
from historical_forecast import get_historical_forecast
from current_forecast import get_current_forecast
from models import TripContext
from display import display, plot_forecast

VALID_METHODS  = ("forecast", "current", "historical")
MAX_FORECAST_DAYS = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Travel Weather Advisor — weather forecast by city and date range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--city",   required=True, help="Destination city, e.g. 'Tokyo' or 'Paris, France'")
    parser.add_argument("--start",  required=True, help="Travel start date (YYYY-MM-DD)")
    parser.add_argument("--end",    required=True, help="Travel end date (YYYY-MM-DD)")
    parser.add_argument("--method", default="forecast", choices=VALID_METHODS,
                        help=(
                            "forecast   — Open-Meteo live model (default, ≤16 days ahead)\n"
                            "current    — custom: recent observed data + anomaly decay\n"
                            "historical — custom: past-years climatology + trend (any date)"
                        ))
    parser.add_argument("--years",  type=int, default=10, metavar="N",
                        help="[historical] years of archive data to use (default: 10)")
    parser.add_argument("--chart",  action="store_true",
                        help="Save a matplotlib forecast chart as PNG")
    return parser.parse_args()


def validate_dates_forecast(start_str, end_str):
    try:
        start = date.fromisoformat(start_str)
        end   = date.fromisoformat(end_str)
    except ValueError as e:
        sys.exit(f"Error: {e}. Use YYYY-MM-DD format.")
    if end < start:
        sys.exit("Error: End date must be on or after start date.")
    today = date.today()
    future_limit = today + timedelta(days=MAX_FORECAST_DAYS - 1)
    if start > future_limit:
        sys.exit(
            f"Error: '{start_str}' is beyond the {MAX_FORECAST_DAYS}-day forecast window "
            f"(max: {future_limit}).\nTip: use --method historical for trips further ahead."
        )
    if end > future_limit:
        print(f"[Warning] End date clamped to forecast limit ({future_limit}).")
        end = future_limit
    return start.isoformat(), end.isoformat()


def validate_dates_historical(start_str, end_str):
    try:
        start = date.fromisoformat(start_str)
        end   = date.fromisoformat(end_str)
    except ValueError as e:
        sys.exit(f"Error: {e}. Use YYYY-MM-DD format.")
    if end < start:
        sys.exit("Error: End date must be on or after start date.")
    if (end - start).days + 1 > 60:
        sys.exit("Error: Historical prediction supports up to 60 days per query.")
    if start.year < 1950:
        sys.exit("Error: Historical data only available from 1950 onwards.")
    return start.isoformat(), end.isoformat()


def main():
    args = parse_args()

    # ── Date validation ────────────────────────────────────────────────────────
    if args.method == "historical":
        start_date, end_date = validate_dates_historical(args.start, args.end)
    else:
        start_date, end_date = validate_dates_forecast(args.start, args.end)

    # ── Geocoding ──────────────────────────────────────────────────────────────
    print(f"Looking up location for '{args.city}'...")
    try:
        loc = get_location(args.city)
    except (ValueError, ConnectionError) as e:
        sys.exit(f"Error: {e}")
    print(f"Found: {loc['name']}, {loc['country']} ({loc['latitude']:.2f}, {loc['longitude']:.2f})")

    # ── Weather forecast ───────────────────────────────────────────────────────
    try:
        if args.method == "historical":
            print(f"Running historical prediction for {start_date} → {end_date}...")
            forecasts = get_historical_forecast(
                lat=loc["latitude"], lon=loc["longitude"], timezone=loc["timezone"],
                start_date=start_date, end_date=end_date, n_years=args.years,
            )
        elif args.method == "current":
            print(f"Running current-conditions prediction for {start_date} → {end_date}...")
            forecasts = get_current_forecast(
                lat=loc["latitude"], lon=loc["longitude"], timezone=loc["timezone"],
                start_date=start_date, end_date=end_date,
            )
        else:
            print(f"Fetching live forecast for {start_date} → {end_date}...")
            forecasts = get_forecast(
                lat=loc["latitude"], lon=loc["longitude"], timezone=loc["timezone"],
                start_date=start_date, end_date=end_date,
            )
    except (ValueError, ConnectionError) as e:
        sys.exit(f"Error: {e}")

    context = TripContext(purpose="", city=loc["name"], country=loc["country"])

    # ── Display ────────────────────────────────────────────────────────────────
    display(forecasts, context, start_date, end_date,
            method=args.method, n_years=args.years if args.method == "historical" else None)

    # ── Chart ──────────────────────────────────────────────────────────────────
    if args.chart:
        chart_path = plot_forecast(forecasts, context, start_date, end_date,
                                   method=args.method, n_years=args.years if args.method == "historical" else None)
        print(f"\nChart saved → {chart_path}")


if __name__ == "__main__":
    main()
