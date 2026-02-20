# portfolio_config.py

# =========================
# ASSET UNIVERSE
# =========================

assets = [

    # =========================
    # LARGE CAP EQUITY
    # =========================
    'RELIANCE.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'INFY.NS',
    'ICICIBANK.NS',
    'HINDUNILVR.NS',
    'LT.NS',
    'SBIN.NS',

    # =========================
    # MID CAP
    # =========================
    'BAJFINANCE.NS',
    'DMART.NS',
    'PIDILITIND.NS',
    'ADANIPORTS.NS',

    # =========================
    # SMALL CAP
    # =========================
    'DEEPAKNTR.NS',
    'NAVINFLUOR.NS',
    'BALKRISIND.NS',

    # =========================
    # INDEX ETFs
    # =========================
    'NIFTYBEES.NS',
    'BANKBEES.NS',
    'MIDCAPETF.NS',

    # =========================
    # SECTOR ETFs
    # =========================
    'ITBEES.NS',
    'PHARMABEES.NS',
    'AUTOIETF.NS',

    # =========================
    # INTERNATIONAL ETFs
    # =========================
    'MON100.NS',      # Nasdaq 100
    'MAFANG.NS',      # FAANG ETF
    'SP500ETF.NS',    # S&P 500 exposure

    # =========================
    # GOLD & SILVER
    # =========================
    'GOLDSHARE.NS',
    'SILVERBEES.NS',

    # =========================
    # DEBT ETFs
    # =========================
    'LIQUIDBEES.NS',
    'GILT5YBEES.NS',
    'CPSEETF.NS',
    'BHARATBOND.NS',

]

# =========================
# ASSET CLASS MAPPING
# =========================

asset_classes = {

    # Equity - Large
    'RELIANCE.NS': 'Equity',
    'TCS.NS': 'Equity',
    'HDFCBANK.NS': 'Equity',
    'INFY.NS': 'Equity',
    'ICICIBANK.NS': 'Equity',
    'HINDUNILVR.NS': 'Equity',
    'LT.NS': 'Equity',
    'SBIN.NS': 'Equity',

    # Equity - Mid
    'BAJFINANCE.NS': 'Equity',
    'DMART.NS': 'Equity',
    'PIDILITIND.NS': 'Equity',
    'ADANIPORTS.NS': 'Equity',

    # Equity - Small
    'DEEPAKNTR.NS': 'Equity',
    'NAVINFLUOR.NS': 'Equity',
    'BALKRISIND.NS': 'Equity',

    # ETFs - Equity
    'NIFTYBEES.NS': 'Equity',
    'BANKBEES.NS': 'Equity',
    'MIDCAPETF.NS': 'Equity',
    'ITBEES.NS': 'Equity',
    'PHARMABEES.NS': 'Equity',
    'AUTOIETF.NS': 'Equity',

    # International
    'MON100.NS': 'International',
    'MAFANG.NS': 'International',
    'SP500ETF.NS': 'International',

    # Gold & Silver
    'GOLDSHARE.NS': 'Gold',
    'SILVERBEES.NS': 'Gold',

    # Debt
    'GILT5YBEES.NS': 'Debt',
    'CPSEETF.NS': 'Debt',
    'BHARATBOND.NS': 'Debt',

    # Cash
    'LIQUIDBEES.NS': 'Cash',
}

constraints_data = [

    # =========================
    # EQUITY LIMIT
    # =========================
    [False, 'Classes', 'Class', 'Equity', '<=', 0.60, '', '', '', ''],

    # =========================
    # INTERNATIONAL LIMIT
    # =========================
    [False, 'Classes', 'Class', 'International', '<=', 0.25, '', '', '', ''],

    # =========================
    # GOLD MINIMUM
    # =========================
    [False, 'Classes', 'Class', 'Gold', '>=', 0.05, '', '', '', ''],
    [False, 'Classes', 'Class', 'Gold', '<=', 0.20, '', '', '', ''],

    # =========================
    # DEBT MINIMUM
    # =========================
    [False, 'Classes', 'Class', 'Debt', '>=', 0.10, '', '', '', ''],

    # =========================
    # CASH BUFFER
    # =========================
    [False, 'Classes', 'Class', 'Cash', '>=', 0.05, '', '', '', ''],
    [False, 'Classes', 'Class', 'Cash', '<=', 0.20, '', '', '', ''],

]