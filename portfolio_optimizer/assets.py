assets = [

    # ---------------- LARGE CAP EQUITY ----------------
    'RELIANCE.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'INFY.NS',
    'ICICIBANK.NS',
    'HINDUNILVR.NS',
    'LT.NS',
    'SBIN.NS',
    'AXISBANK.NS',
    'KOTAKBANK.NS',
    'ITC.NS',
    'BHARTIARTL.NS',
    'ASIANPAINT.NS',
    'MARUTI.NS',
    'SUNPHARMA.NS',

    # ---------------- MID CAP ----------------
    'BAJFINANCE.NS',
    'DMART.NS',
    'PIDILITIND.NS',
    'ADANIPORTS.NS',
    'TRENT.NS',
    'PAGEIND.NS',
    'MUTHOOTFIN.NS',
    'SRF.NS',
    'POLYCAB.NS',
    'ABB.NS',

    # ---------------- SMALL CAP ----------------
    'DEEPAKNTR.NS',
    'NAVINFLUOR.NS',
    'BALKRISIND.NS',
    'ASTRAL.NS',
    'KEI.NS',
    'TANLA.NS',
    'CLEAN.NS',
    'ROUTE.NS',

    # ---------------- INDEX ETFs ----------------
    'NIFTYBEES.NS',
    'BANKBEES.NS',
    'MOM100.NS',         
    'JUNIORBEES.NS',
    
    # ---------------- SECTOR ETFs ----------------
    'ITBEES.NS',
    'PHARMABEES.NS',     
    'AUTOIETF.NS',       
    'PSUBNKBEES.NS',

    # ---------------- PSU EQUITY ETF ----------------
    'CPSEETF.NS',

    # ---------------- INTERNATIONAL ETFs ----------------
    'MON100.NS',        
    'MAFANG.NS',         
    'ICICIB22.NS',       

    # ---------------- COMMODITIES ----------------
   
    'GOLDBEES.NS',
    'SILVERBEES.NS',     # listed ~Jan 2022 — use start="2022-01-01"

    # ---------------- DEBT ETFs ----------------
    'LIQUIDBEES.NS',
    'GILT5YBEES.NS',
    'EBBETF0431.NS',     
    'SDL26BEES.NS',

    # ---------------- REITS ----------------
    'EMBASSY.NS',
    'MINDSPACE.NS',
    'BIRET.NS',          

    # ---------------- INVITS ----------------
    'INDIGRID.NS',
    'IRBINVIT.NS',
    'PGINVIT.NS',

]

# ASSET CLASS MAPPING

asset_classes = {

    # ---- Large Cap Equity ----
    'RELIANCE.NS':    'Equity',
    'TCS.NS':         'Equity',
    'HDFCBANK.NS':    'Equity',
    'INFY.NS':        'Equity',
    'ICICIBANK.NS':   'Equity',
    'HINDUNILVR.NS':  'Equity',
    'LT.NS':          'Equity',
    'SBIN.NS':        'Equity',
    'AXISBANK.NS':    'Equity',
    'KOTAKBANK.NS':   'Equity',
    'ITC.NS':         'Equity',
    'BHARTIARTL.NS':  'Equity',
    'ASIANPAINT.NS':  'Equity',
    'MARUTI.NS':      'Equity',
    'SUNPHARMA.NS':   'Equity',

    # ---- Mid Cap Equity ----
    'BAJFINANCE.NS':  'Equity',
    'DMART.NS':       'Equity',
    'PIDILITIND.NS':  'Equity',
    'ADANIPORTS.NS':  'Equity',
    'TRENT.NS':       'Equity',
    'PAGEIND.NS':     'Equity',
    'MUTHOOTFIN.NS':  'Equity',
    'SRF.NS':         'Equity',
    'POLYCAB.NS':     'Equity',
    'ABB.NS':         'Equity',

    # ---- Small Cap Equity ----
    'DEEPAKNTR.NS':   'Equity',
    'NAVINFLUOR.NS':  'Equity',
    'BALKRISIND.NS':  'Equity',
    'ASTRAL.NS':      'Equity',
    'KEI.NS':         'Equity',
    'TANLA.NS':       'Equity',
    'CLEAN.NS':       'Equity',
    'ROUTE.NS':       'Equity',

    # ---- Index ETFs ----
    'NIFTYBEES.NS':   'Equity',
    'BANKBEES.NS':    'Equity',
    'MOM100.NS':      'Equity',    
    'JUNIORBEES.NS':  'Equity',
    

    # ---- Sector ETFs ----
    'ITBEES.NS':      'Equity',
    'PHARMABEES.NS':  'Equity',
    'AUTOIETF.NS':    'Equity',
    'PSUBNKBEES.NS':  'Equity',
    

    # ---- PSU Equity ETF ----
    'CPSEETF.NS':     'Equity',

    # ---- International ETFs ----
    'MON100.NS':      'International',
    'MAFANG.NS':      'International',
    'ICICIB22.NS':    'International',

    # ---- Gold ----
    'GOLDBEES.NS':    'Gold',
    'SILVERBEES.NS':  'Gold',

    # ---- Debt ----
    'GILT5YBEES.NS':  'Debt',
    'EBBETF0431.NS':  'Debt',      
    'SDL26BEES.NS':   'Debt',

    # ---- Cash ----
    'LIQUIDBEES.NS':  'Cash',

    # ---- Real Estate (REITs) ----
    'EMBASSY.NS':     'RealEstate',
    'MINDSPACE.NS':   'RealEstate',
    'BIRET.NS':       'RealEstate',    

    # ---- Infrastructure (InvITs) ----
    'INDIGRID.NS':    'Infra',
    'IRBINVIT.NS':    'Infra',
    'PGINVIT.NS':     'Infra',

}

constraints_data = [

    # EQUITY — floor added to ensure minimum growth allocation
    [False, 'Classes', 'Class', 'Equity',        '>=', 0.20, '', '', '', ''],
    [False, 'Classes', 'Class', 'Equity',        '<=', 0.60, '', '', '', ''],

    # INTERNATIONAL — floor added to ensure global exposure
    [False, 'Classes', 'Class', 'International', '>=', 0.05, '', '', '', ''],
    [False, 'Classes', 'Class', 'International', '<=', 0.25, '', '', '', ''],

    # GOLD
    [False, 'Classes', 'Class', 'Gold',          '>=', 0.05, '', '', '', ''],
    [False, 'Classes', 'Class', 'Gold',          '<=', 0.20, '', '', '', ''],

    # DEBT — ceiling added to prevent over-allocation
    [False, 'Classes', 'Class', 'Debt',          '>=', 0.10, '', '', '', ''],
    [False, 'Classes', 'Class', 'Debt',          '<=', 0.40, '', '', '', ''],

    # REAL ESTATE
    [False, 'Classes', 'Class', 'RealEstate',    '<=', 0.10, '', '', '', ''],

    # INFRA
    [False, 'Classes', 'Class', 'Infra',         '<=', 0.10, '', '', '', ''],

    # CASH
    [False, 'Classes', 'Class', 'Cash',          '>=', 0.05, '', '', '', ''],
    [False, 'Classes', 'Class', 'Cash',          '<=', 0.20, '', '', '', ''],

]

# ── CONSTRAINTS SUMMARY ───────────────────────────────────────────────────────
# Class          Floor    Ceiling
# Equity          20%      60%
# International    5%      25%
# Gold             5%      20%
# Debt            10%      40%
# RealEstate       —       10%
# Infra            —       10%
# Cash             5%      20%
#
# Floor sum  = 20+5+5+10+5 = 45%  (<100% ✓ feasible)
# Ceiling sum = 60+25+20+40+10+10+20 = 185%  (>100% ✓ feasible)
# ─────────────────────────────────────────────────────────────────────────────


# ── SUMMARY ──────────────────────────────────────────────────────────────────
# Total instruments : 56
# Symbol fixes      : MIDCAPETF.NS  → MOM100.NS      (Motilal Midcap 100)
#                     EBBETF0433.NS → EBBETF0431.NS   (Bharat Bond Apr 2031)
# Removed           : FMCGETF.NS    (no valid Yahoo symbol; FMCG covered by stocks)
#
# Class breakdown:
#   Equity          : 37  (stocks + index ETFs + sector ETFs + CPSEETF)
#   International   :  3  (MON100, MAFANG, ICICIB22)
#   Gold            :  2  (GOLDBEES, SILVERBEES)
#   Debt            :  3  (GILT5YBEES, EBBETF0431, SDL26BEES)
#   Cash            :  1  (LIQUIDBEES)
#   RealEstate      :  3  (EMBASSY, MINDSPACE, BIRET)
#   Infra           :  3  (INDIGRID, IRBINVIT, PGINVIT)
# ─────────────────────────────────────────────────────────────────────────────


def _validate():
    # 1. Duplicates in assets list
    seen = set()
    dupes = []
    for t in assets:
        if t in seen:
            dupes.append(t)
        seen.add(t)
    if dupes:
        raise ValueError(
            f"[assets.py] Duplicate tickers in assets list — "
            f"this corrupts the covariance matrix:\n  {dupes}"
        )

    # 2. In assets but not mapped in asset_classes
    missing_class = [t for t in assets if t not in asset_classes]
    if missing_class:
        raise ValueError(
            f"[assets.py] Tickers in assets list with no asset_classes entry:\n"
            f"  {missing_class}"
        )

    # 3. In asset_classes but not in assets list
    missing_asset = [t for t in asset_classes if t not in assets]
    if missing_asset:
        raise ValueError(
            f"[assets.py] Tickers in asset_classes but missing from assets list:\n"
            f"  {missing_asset}"
        )

    print(f"[assets.py] Validation passed - "
          f"{len(assets)} unique tickers across "
          f"{len(set(asset_classes.values()))} asset classes.")

_validate()
# ─────────────────────────────────────────────────────────────────────────────
