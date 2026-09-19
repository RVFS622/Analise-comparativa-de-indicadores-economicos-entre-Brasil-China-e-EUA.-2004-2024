from __future__ import annotations


from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
ANALYSIS_START_YEAR = 2004
ANALYSIS_END_YEAR = 2024
WID_GROUPS = ["p0p50", "p50p90", "p90p100", "p99p100"]

COUNTRY_CODE_MAP = {"BR": "BRA", "BRA": "BRA", "US": "USA", "USA": "USA", "CN": "CHN", "CHN": "CHN"}

SECTOR_INDICATORS = {
    "SL.AGR.EMPL.ZS": "Agricultura",
    "SL.IND.EMPL.ZS": "Indústria",
    "SL.SRV.EMPL.ZS": "Serviços",
}


def load_world_bank_gdp() -> pd.DataFrame:
    path = ROOT / "API_NY.GDP.PCAP.KD_DS2_en_csv_v2_329799" / "API_NY.GDP.PCAP.KD_DS2_en_csv_v2_329799.csv"
    df = pd.read_csv(path, skiprows=4, engine="python")
    year_cols = [c for c in df.columns if c.isdigit()]
    df = df[["Country Code", "Indicator Code", *year_cols]].copy()
    df = df[df["Indicator Code"] == "NY.GDP.PCAP.KD"].copy()
    df = df.rename(columns={"Country Code": "country"})
    long_df = df.melt(id_vars=["country", "Indicator Code"], value_vars=year_cols, var_name="year", value_name="obs_value")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["obs_value"] = pd.to_numeric(long_df["obs_value"], errors="coerce")
    return long_df[["country", "year", "obs_value"]].dropna(subset=["country", "year", "obs_value"]).sort_values(["country", "year"]).reset_index(drop=True)


def load_ilostat_series() -> pd.DataFrame:
    files = {
        "gdp_211p": ROOT / "GDP_211P_NOC_NB_A-20260919T1910.csv" / "GDP_211P_NOC_NB_A-20260919T1910.csv",
        "sdg_1041": ROOT / "SDG_1041_NOC_RT_A-20260919T1910.csv" / "SDG_1041_NOC_RT_A-20260919T1910.csv",
    }

    records = []
    for name, path in files.items():
        df = pd.read_csv(path)
        df = df.rename(columns={"ref_area": "country", "time": "year", "obs_value": "value"})
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["indicator"] = name
        records.append(df[["country", "year", "indicator", "value"]])

    return pd.concat(records, ignore_index=True).dropna(subset=["country", "year", "value"]).sort_values(["country", "year"]).reset_index(drop=True)


def load_world_bank_sector_employment() -> pd.DataFrame:
    path = ROOT / "world_bank_sector_employment"
    records = []
    for indicator, sector in SECTOR_INDICATORS.items():
        file_path = path / f"{indicator}.csv"
        df = pd.read_csv(file_path)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["sector"] = sector
        records.append(df[["country", "country_name", "year", "value", "sector"]])

    return (
        pd.concat(records, ignore_index=True)
        .dropna(subset=["country", "year", "value"])
        .sort_values(["country", "year", "sector"])
        .reset_index(drop=True)
    )


def load_wid_country(country_code: str = "BR") -> pd.DataFrame:
    path = ROOT / "wid_all_data" / f"WID_data_{country_code}.csv"
    df = pd.read_csv(path, sep=";")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["year", "value"]).sort_values(["variable", "year"]).reset_index(drop=True)


def calc_growth_pct(start_value: float, end_value: float) -> float:
    if pd.isna(start_value) or pd.isna(end_value) or start_value == 0:
        return float("nan")
    return ((end_value / start_value) - 1.0) * 100.0


def normalize_country_code(country_code: str) -> str:
    return COUNTRY_CODE_MAP.get(country_code.upper(), country_code.upper())


def summarize_country(country_code: str = "BR") -> None:
    wb_code = normalize_country_code(country_code)
    wid_code = country_code.upper()

    print(f"\n=== Análise para {country_code} ===")

    wb = load_world_bank_gdp()
    wb_country = wb[
        (wb["country"] == wb_code)
        & wb["year"].between(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR)
    ].sort_values("year")
    if wb_country.empty:
        print(f"Não há dados de PIB per capita para o código {wb_code} no World Bank.")
        return

    start_year = int(wb_country["year"].min())
    latest_year = int(wb_country["year"].max())
    start_value = wb_country.loc[wb_country["year"] == start_year, "obs_value"].iloc[0]
    latest_value = wb_country.loc[wb_country["year"] == latest_year, "obs_value"].iloc[0]
    growth = calc_growth_pct(start_value, latest_value)
    print("PIB per capita (World Bank):")
    print(f"- Ano inicial: {start_year}, valor: {start_value:,.2f}")
    print(f"- Ano final: {latest_year}, valor: {latest_value:,.2f}")
    print(f"- Crescimento acumulado: {growth:.2f}%")

    ilostat = load_ilostat_series()
    ilostat_country = ilostat[
        (ilostat["country"] == wb_code)
        & ilostat["year"].between(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR)
    ].pivot(index="year", columns="indicator", values="value")
    if not ilostat_country.empty:
        print("\nIndicadores ILOSTAT:")
        print(ilostat_country.tail(5).round(2).to_string())

    wid = load_wid_country(wid_code)
    top_income = wid[
        (wid["variable"] == "aptincj992")
        & wid["percentile"].isin(WID_GROUPS)
        & wid["year"].between(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR)
    ].copy()
    if not top_income.empty:
        income_pivot = top_income.pivot(index="year", columns="percentile", values="value")
        wid_target_year = int(income_pivot.index.max())
        print("\nDistribuição de renda (WID) — valores por percentil:")
        print(income_pivot.loc[sorted(income_pivot.index)[-5:]].tail(5).round(4).to_string())

        for pct in ["p0p50", "p50p90", "p90p100", "p99p100"]:
            if pct in income_pivot.columns and wid_target_year in income_pivot.index:
                val = income_pivot.loc[wid_target_year, pct]
                print(f"- Renda anual média {pct} em {wid_target_year}: {val:,.2f}")

    wealth = wid[
        (wid["variable"] == "ahwealj992")
        & wid["percentile"].isin(WID_GROUPS)
        & wid["year"].between(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR)
    ].copy()
    if not wealth.empty:
        wealth_pivot = wealth.pivot(index="year", columns="percentile", values="value")
        print("\nDistribuição de riqueza (WID):")
        print(wealth_pivot.loc[sorted(wealth_pivot.index)[-5:]].tail(5).round(4).to_string())

    print("\nResumo operacional:")
    print("- Cálculo de crescimento do PIB per capita realizado.")
    print("- Séries de produtividade e participação do trabalho carregadas do ILOSTAT.")
    print("- Distribuição de renda/riqueza do WID validada para percentis relevantes.")


def summarize_sector_employment() -> None:
    sectors = load_world_bank_sector_employment()
    latest = sectors.groupby(["country", "country_name", "sector"], as_index=False)["year"].max()
    latest = latest.merge(sectors, on=["country", "country_name", "sector", "year"])
    print("\nEmprego por setor (World Bank):")
    print(latest.sort_values(["country_name", "sector"])[["country_name", "year", "sector", "value"]].round(2).to_string(index=False))


def main() -> None:
    summarize_country("BR")
    summarize_sector_employment()


if __name__ == "__main__":
    main()
