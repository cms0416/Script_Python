#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


@dataclass
class Config:
    sheet_name: str
    fx_source: str  # "sheet" or "override"
    fx_override: Dict[str, float]  # like {"USD": 1380.5}
    worksheet_transactions: str = "Transactions"
    ws_positions: str = "Positions"
    ws_trades_pnl: str = "Trades_PnL"
    ws_monthly_pnl: str = "Monthly_PnL"


def auth_gspread() -> gspread.Client:
    # Uses GOOGLE_APPLICATION_CREDENTIALS environment variable for service account key
    creds = Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=SCOPES,
    )
    gc = gspread.authorize(creds)
    return gc


def open_sheet(gc: gspread.Client, sheet_name: str) -> gspread.Spreadsheet:
    return gc.open(sheet_name)


def read_transactions(sh: gspread.Spreadsheet, ws_name: str) -> pd.DataFrame:
    try:
        ws = sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        raise RuntimeError(f"Worksheet '{ws_name}' not found.")
    rows = ws.get_all_records()
    df = pd.DataFrame(rows)

    # Normalize columns
    required = ["Date", "Account", "Ticker", "Market", "Side", "Qty", "Price", "Fee", "Tax", "Currency"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # Parse
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    num_cols = ["Qty", "Price", "Fee", "Tax"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["Side"] = df["Side"].str.upper().str.strip()
    df["Currency"] = df["Currency"].str.upper().str.strip().fillna("KRW")

    # Optional per-row FX column
    if "FX" in df.columns:
        df["FX"] = pd.to_numeric(df["FX"], errors="coerce")
    else:
        df["FX"] = np.nan

    # Keep only valid rows with Date and Ticker and Side in BUY/SELL/DIV
    df = df[df["Date"].notna() & df["Ticker"].notna() & df["Side"].isin(["BUY", "SELL", "DIV"])].copy()
    df.sort_values(["Account", "Ticker", "Date"], inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)
    return df


def read_named_fx(ws: gspread.Worksheet, code: str) -> float:
    # Try to read named range like FX_USD
    try:
        rng = ws.spreadsheet.fetch_sheet_metadata()
        named = rng.get("namedRanges", [])
        target = None
        for n in named:
            if n["name"] == f"FX_{code}":
                target = n["range"]
                break
        if not target:
            return np.nan
        # Extract actual values
        sh = ws.spreadsheet
        w = sh.get_worksheet_by_id(target["sheetId"])
        a1 = gspread.utils.rowcol_to_a1(target["startRowIndex"] + 1, target["startColumnIndex"] + 1)
        v = w.acell(a1).value
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan


def resolve_fx_per_row(df: pd.DataFrame, sh: gspread.Spreadsheet, cfg: Config) -> pd.Series:
    # Priority: row FX -> override -> named range -> 1 for KRW
    base_ws = sh.sheet1  # use first sheet to host named ranges, or change here
    out = []
    cache_named: Dict[str, float] = {}
    for _, r in df.iterrows():
        cur = r["Currency"]
        if cur == "KRW":
            out.append(1.0)
            continue
        if not np.isnan(r["FX"]):
            out.append(float(r["FX"]))
            continue
        if cfg.fx_source == "override" and cur in cfg.fx_override:
            out.append(cfg.fx_override[cur])
            continue
        # named range
        if cur not in cache_named:
            cache_named[cur] = read_named_fx(base_ws, cur)
        val = cache_named[cur]
        out.append(val if not np.isnan(val) else np.nan)
    return pd.to_numeric(pd.Series(out), errors="coerce")


def fifo_pnl(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute FIFO per (Account, Ticker). Returns (positions, trades_pnl).
    - Positions: remaining lots with avg cost, qty, MV (not computed here), and cost in KRW
    - Trades_PnL: realized PnL per sell trade (both in trade currency and KRW)
    """
    records_trades = []
    records_pos = []

    for (acct, tic), g in df.groupby(["Account", "Ticker"], sort=False):
        lots: List[Dict] = []  # each: {"qty": float, "price": float, "fee": float, "tax": float, "fx": float, "date": Timestamp}
        currency = g["Currency"].iloc[0]

        for _, r in g.iterrows():
            side = r["Side"]
            qty = r["Qty"]
            price = r["Price"]
            fee = r["Fee"]
            tax = r["Tax"]
            fx = r.get("FX_resolved", np.nan)
            date = r["Date"]

            if side == "BUY":
                lots.append({"qty": qty, "price": price, "fee": fee, "tax": tax, "fx": fx, "date": date})
            elif side == "DIV":
                # record as separate cashflow; not affecting lots
                records_trades.append({
                    "Date": date, "Account": acct, "Ticker": tic, "Currency": currency,
                    "Type": "DIV", "Qty": 0.0, "Proceeds": r.get("Amount", np.nan) if "Amount" in r else np.nan,
                    "Fee": fee, "Tax": tax, "Fx": fx, "RealizedPnL": 0.0, "RealizedPnL_KRW": 0.0
                })
            elif side == "SELL":
                remain = qty
                proceeds = qty * price - fee - tax
                total_cost = 0.0
                total_cost_krw = 0.0
                total_pnl = 0.0
                total_pnl_krw = 0.0

                while remain > 1e-12 and lots:
                    lot = lots[0]
                    take = min(remain, lot["qty"])
                    cost = take * lot["price"]
                    # allocate proportional fees on buy; conservative: ignore or include in cost
                    cost += 0.0  # could add lot["fee"] * (take / lot["qty"]) if desired
                    pnl = take * (price - lot["price"])
                    # FX: convert to KRW using per-row resolved FX; if missing, leave NaN
                    fx_sell = fx
                    fx_buy = lot["fx"]
                    cost_krw = np.nan
                    pnl_krw = np.nan
                    if not np.isnan(fx_sell) and not np.isnan(fx_buy):
                        # Using sell FX for proceeds, buy FX for cost (closer to realized basis in KRW)
                        cost_krw = take * lot["price"] * fx_buy
                        pnl_krw = take * (price * fx_sell - lot["price"] * fx_buy)
                    total_cost += cost
                    if not np.isnan(cost_krw):
                        total_cost_krw += cost_krw
                    total_pnl += pnl
                    if not np.isnan(pnl_krw):
                        total_pnl_krw += pnl_krw

                    lot["qty"] -= take
                    remain -= take
                    if lot["qty"] <= 1e-12:
                        lots.pop(0)

                records_trades.append({
                    "Date": date, "Account": acct, "Ticker": tic, "Currency": currency,
                    "Type": "SELL", "Qty": qty, "Proceeds": proceeds, "Fee": fee, "Tax": tax, "Fx": fx,
                    "RealizedPnL": total_pnl, "RealizedPnL_KRW": total_pnl_krw if total_pnl_krw != 0.0 else np.nan
                })

        # after processing, record remaining position summary
        rem_qty = sum(l["qty"] for l in lots)
        if rem_qty > 1e-12:
            # avg cost in trade currency (simple weighted)
            wcost = sum(l["qty"] * l["price"] for l in lots) / rem_qty
            # avg FX cost (weighted by qty*price)
            fx_vals = [l["fx"] for l in lots if not np.isnan(l["fx"])]
            if fx_vals:
                wfx = sum(l["qty"] * l["price"] * l["fx"] for l in lots if not np.isnan(l["fx"])) / \
                      sum(l["qty"] * l["price"] for l in lots if not np.isnan(l["fx"]))
            else:
                wfx = np.nan
            records_pos.append({
                "Account": acct, "Ticker": tic, "Currency": currency,
                "Qty": rem_qty, "AvgCost": wcost, "AvgCost_KRW": wcost * wfx if not np.isnan(wfx) else np.nan
            })

    pos = pd.DataFrame.from_records(records_pos) if records_pos else pd.DataFrame(columns=["Account","Ticker","Currency","Qty","AvgCost","AvgCost_KRW"])
    trades = pd.DataFrame.from_records(records_trades) if records_trades else pd.DataFrame(columns=["Date","Account","Ticker","Currency","Type","Qty","Proceeds","Fee","Tax","Fx","RealizedPnL","RealizedPnL_KRW"])
    if not trades.empty:
        trades.sort_values("Date", inplace=True, kind="stable")
    return pos, trades


def monthly_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["YearMonth", "Account", "RealizedPnL", "RealizedPnL_KRW"])
    t = trades[trades["Type"] == "SELL"].copy()
    t["YearMonth"] = t["Date"].dt.to_period("M").astype(str)
    out = t.groupby(["YearMonth", "Account"], as_index=False)[["RealizedPnL", "RealizedPnL_KRW"]].sum(min_count=1)
    return out


def write_sheet(sh: gspread.Spreadsheet, name: str, df: pd.DataFrame):
    try:
        ws = sh.worksheet(name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=100, cols=26)
    set_with_dataframe(ws, df.fillna(""), include_index=False, include_column_header=True, resize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True, help="Google Sheet name")
    ap.add_argument("--fx_source", choices=["sheet", "override"], default="sheet")
    ap.add_argument("--fx_override", nargs="*", default=[], help="Pairs like USD=1380.5 JPY=9.2")
    args = ap.parse_args()

    fx_map = {}
    for p in args.fx_override:
        k, v = p.split("=")
        fx_map[k.upper()] = float(v)

    cfg = Config(sheet_name=args.sheet, fx_source=args.fx_source, fx_override=fx_map)

    gc = auth_gspread()
    sh = open_sheet(gc, cfg.sheet_name)

    tdf = read_transactions(sh, cfg.worksheet_transactions)

    # Resolve FX per row
    tdf["FX_resolved"] = resolve_fx_per_row(tdf, sh, cfg)

    # FIFO
    pos, trades = fifo_pnl(tdf)

    # Monthly PnL
    monthly = monthly_pnl(trades)

    # Write back
    write_sheet(sh, cfg.ws_positions, pos)
    write_sheet(sh, cfg.ws_trades_pnl, trades)
    write_sheet(sh, cfg.ws_monthly_pnl, monthly)

    print("Done.")


if __name__ == "__main__":
    main()
