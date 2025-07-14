# -*- coding: utf-8 -*-
"""
Aplikacja Streamlit do Prognozowania Szeregów Czasowych.

Umożliwia użytkownikom wgrywanie danych szeregów czasowych,
konfigurowanie modelu (Holt, Winters, Brown) i generowanie
interaktywnej prognozy na przyszłość.

Wersja: Z obsługą błędu niewystarczającej ilości danych dla modeli sezonowych.
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing
import plotly.graph_objects as go
import io
import traceback
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Konfiguracja strony ---
st.set_page_config(
    layout="wide",
    page_title="Prognozowanie Szeregów Czasowych",
    page_icon="📈"
)

# --- Stałe i Słowniki Aplikacji ---
COLUMN_ROLES = ["Ignoruj", "Kolumna Daty", "Kolumna Wartości do Prognozowania"]
DATE_FORMATS = {
    "Wykryj automatycznie": None, "RRRR-MM-DD": "%Y-%m-%d", "DD-MM-RRRR": "%d-%m-%Y",
    "MM-DD-RRRR": "%m-%d-%Y", "RRRR/MM/DD": "%Y/%m/%d", "DD/MM/RRRR": "%d/%m/%Y",
    "MM/DD/RRRR": "%m/%d/%Y", "DD.MM.RRRR": "%d.%m.%Y", "RRRR.MM.DD": "%Y.%m.%d",
    "RRRR-MM": "%Y-%m", "MM-RRRR": "%m-%Y"
}
MODELS = {
    "Model Browna (Podwójne Wygładzanie Wykładnicze)": "brown",
    "Model Holta (Trend Liniowy)": "holt",
    "Model Wintersa (Addytywny)": "winters_add",
    "Model Wintersa (Multiplikatywny)": "winters_mul"
}

# --- Zarządzanie Stanem Sesji ---
def initialize_session_state():
    st.session_state.clear()
    st.session_state.active_step = 1
    st.session_state.df_raw = None
    st.session_state.column_configs = {}
    st.session_state.ts_data = None
    st.session_state.data_is_processed = False
    st.session_state.csv_separator = ','
    st.session_state.csv_encoding = 'utf-8'
    st.session_state.uploaded_file_id = None
    st.session_state.processing_warnings = []
    st.session_state.analysis_generated = False
    st.session_state.use_numeric_index = False

if 'active_step' not in st.session_state:
    initialize_session_state()

# --- Funkcje Pomocnicze ---
@st.cache_data
def parse_uploaded_file(uploaded_file, separator: str, encoding: str) -> pd.DataFrame | None:
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, sep=separator, encoding=encoding, low_memory=False)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Niewspierany format pliku. Proszę użyć CSV lub XLSX.")
            return None
    except Exception as e:
        st.error(f"Błąd podczas wczytywania pliku: {e}")
        return None

# --- Główne Funkcje Przetwarzania i Modelowania ---
@st.cache_data
def process_input_data(_df_raw: pd.DataFrame, _configs: dict, _use_numeric_index: bool) -> tuple:
    df = _df_raw.copy()
    date_col_name, value_col_name = None, None
    warnings = []

    for col, config in _configs.items():
        role = config.get('role', "Ignoruj")
        if role == "Kolumna Daty":
            if date_col_name: continue
            date_col_name = col
            try:
                original_non_nulls = df[col].notna().sum()
                df[col] = pd.to_datetime(df[col].astype(str), format=config.get('format'), errors='coerce')
                converted_nulls = df[col].isnull().sum()
                if converted_nulls > 0:
                    warnings.append(f"W kolumnie daty `{col}` nie udało się przekonwertować {converted_nulls - (len(df) - original_non_nulls)} wartości. Zostały one usunięte.")
                if df[col].isnull().all():
                    st.error(f"Błąd konwersji daty: Wszystkie wartości w kolumnie '{col}' są nieprawidłowe lub puste.")
                    return None, warnings
            except Exception as e:
                st.error(f"Krytyczny błąd konwersji daty w kolumnie '{col}': {e}")
                return None, warnings
        
        elif role == "Kolumna Wartości do Prognozowania":
            if value_col_name: continue
            value_col_name = col
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                warnings.append(f"W kolumnie wartości `{col}` znaleziono i usunięto wiersze z wartościami nienumerycznymi.")

    if not date_col_name and not _use_numeric_index:
        st.error("Musisz zdefiniować 'Kolumnę Daty' lub zaznaczyć opcję utworzenia indeksu numerycznego.")
        return None, warnings
    if not value_col_name:
        st.error("Musisz zdefiniować 'Kolumnę Wartości do Prognozowania'.")
        return None, warnings

    if not date_col_name and _use_numeric_index:
        warnings.append("Utworzono numeryczny indeks porządkowy zamiast kolumny daty.")
        date_col_name = "Indeks_Numeryczny"
        df[date_col_name] = np.arange(len(df))

    df = df[[date_col_name, value_col_name]].dropna()
    df = df.set_index(date_col_name).sort_index()

    if df.index.duplicated().any():
        warnings.append("Wykryto zduplikowane wartości w indeksie (daty lub numery). Agreguję wartości, obliczając średnią.")
        df = df.groupby(df.index).mean()

    if df[value_col_name].isnull().any():
        warnings.append("Wykryto brakujące wartości w danych. Zostaną one uzupełnione metodą 'forward fill'.")
        df[value_col_name] = df[value_col_name].fillna(method='ffill')

    if df.empty:
        st.error("Po przetworzeniu nie pozostały żadne prawidłowe dane do analizy.")
        return None, warnings

    if isinstance(df.index, pd.DatetimeIndex):
        freq = pd.infer_freq(df.index)
        if freq:
            df = df.asfreq(freq)
            warnings.append(f"Automatycznie wykryto częstotliwość danych: {freq}.")

    return df[value_col_name], warnings

def run_forecasting_model(ts_data: pd.Series, model_type: str, forecast_periods: int, seasonal_periods: int | None = None) -> tuple:
    if ts_data.empty:
        raise ValueError("Szereg czasowy do prognozowania jest pusty.")
    
    if model_type == 'holt':
        model = Holt(ts_data, initialization_method="estimated").fit()
    elif model_type == 'brown':
        model = ExponentialSmoothing(ts_data, trend='add', damped_trend=False, initialization_method="estimated").fit()
    elif model_type in ['winters_add', 'winters_mul']:
        if not seasonal_periods or seasonal_periods < 2:
            raise ValueError("Dla modelu Wintersa musisz podać poprawną liczbę okresów sezonowych (np. 12 dla danych miesięcznych).")
        trend_type = 'add'
        seasonal_type = 'add' if model_type == 'winters_add' else 'mul'
        model = ExponentialSmoothing(
            ts_data, trend=trend_type, seasonal=seasonal_type,
            seasonal_periods=seasonal_periods, initialization_method="estimated"
        ).fit()
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

    forecast = model.forecast(forecast_periods)
    fitted_values = model.fittedvalues
    mae = mean_absolute_error(ts_data, fitted_values)
    rmse = np.sqrt(mean_squared_error(ts_data, fitted_values))
    metrics = {"MAE": mae, "RMSE": rmse}

    return model, forecast, metrics

# --- Główne UI ---
st.title("📈 Prognozowanie Szeregów Czasowych")
st.markdown("Witaj! Ta aplikacja pomoże Ci stworzyć prognozę na podstawie historycznych danych, używając modeli wygładzania wykładniczego (Holt, Winters, Brown).")

with st.expander("Krok 1: Wgraj swoje dane", expanded=st.session_state.active_step == 1):
    st.markdown("Wgraj plik (CSV lub XLSX) zawierający Twoje dane. Upewnij się, że masz kolumnę z datami i kolumnę z wartościami numerycznymi, które chcesz prognozować.")
    uploaded_file = st.file_uploader("Wgraj plik z danymi (CSV lub XLSX)", type=["csv", "xlsx"])

    if uploaded_file:
        current_file_id = (uploaded_file.name, uploaded_file.size)
        if current_file_id != st.session_state.get('uploaded_file_id'):
            st.info("Wykryto nowy plik. Resetuję stan aplikacji...")
            initialize_session_state()
            st.session_state.uploaded_file_id = current_file_id
            st.rerun()

        c1, c2 = st.columns(2)
        st.session_state.csv_separator = c1.text_input("Separator dla plików CSV:", value=st.session_state.csv_separator)
        st.session_state.csv_encoding = c2.selectbox("Kodowanie dla plików CSV:", ["utf-8", "latin1", "cp1250"], index=["utf-8", "latin1", "cp1250"].index(st.session_state.csv_encoding))

        df_raw = parse_uploaded_file(uploaded_file, st.session_state.csv_separator, st.session_state.csv_encoding)

        if df_raw is not None:
            st.session_state.df_raw = df_raw
            if not st.session_state.column_configs:
                st.session_state.column_configs = {c: {'role': "Ignoruj"} for c in df_raw.columns}

            st.dataframe(df_raw.head(), use_container_width=True)
            if st.button("Przejdź do konfiguracji kolumn", type="primary"):
                st.session_state.active_step = 2
                st.rerun()

with st.expander("Krok 2: Skonfiguruj dane", expanded=st.session_state.active_step == 2):
    if st.session_state.get('df_raw') is not None:
        st.subheader("Zdefiniuj role dla kluczowych kolumn")
        st.info("Wskaż, która kolumna zawiera daty, a która wartości do prognozowania. Jeśli nie masz kolumny z datami, możesz skorzystać z opcji utworzenia indeksu numerycznego poniżej.")

        if st.session_state.processing_warnings:
            for warning_msg in st.session_state.processing_warnings:
                st.warning(warning_msg, icon="⚠️")
            st.session_state.processing_warnings = []

        cols_ui = st.columns(3)
        for i, col_name in enumerate(st.session_state.column_configs.keys()):
            with cols_ui[i % 3]:
                st.markdown(f"**`{col_name}`**")
                cfg = st.session_state.column_configs[col_name]
                current_role = cfg.get('role', "Ignoruj")
                cfg['role'] = st.selectbox("Rola:", COLUMN_ROLES, index=COLUMN_ROLES.index(current_role), key=f"role_{col_name}")
                if cfg['role'] == "Kolumna Daty":
                    current_fmt_key = next((k for k, v in DATE_FORMATS.items() if v == cfg.get('format')), "Wykryj automatycznie")
                    cfg['format'] = DATE_FORMATS[st.selectbox("Format Daty:", list(DATE_FORMATS.keys()), index=list(DATE_FORMATS.keys()).index(current_fmt_key), key=f"format_{col_name}")]

        is_date_col_selected = any(c.get('role') == "Kolumna Daty" for c in st.session_state.column_configs.values())
        if not is_date_col_selected:
            st.session_state.use_numeric_index = st.checkbox(
                "Nie mam kolumny daty - utwórz indeks numeryczny", 
                value=st.session_state.use_numeric_index,
                help="Zaznacz, jeśli Twoje dane nie mają kolumny z datą. Aplikacja użyje kolejnych numerów wierszy (0, 1, 2...) jako osi czasu."
            )
        else:
            st.session_state.use_numeric_index = False

        if st.button("Zatwierdź i przejdź do prognozy", type="primary"):
            ts_data, warnings = process_input_data(st.session_state.df_raw, st.session_state.column_configs, st.session_state.use_numeric_index)
            st.session_state.processing_warnings = warnings
            if ts_data is not None:
                st.session_state.ts_data = ts_data
                st.session_state.data_is_processed = True
                st.session_state.active_step = 3
                st.session_state.analysis_generated = False
                st.success("Dane zostały pomyślnie przetworzone!")
                st.rerun()
    else:
        st.warning("Wróć do Kroku 1 i wczytaj plik z danymi.")


with st.expander("Krok 3: Wygeneruj i analizuj prognozę", expanded=st.session_state.active_step == 3):
    if st.session_state.get('data_is_processed'):
        st.subheader("Konfiguracja modelu prognozowania")
        
        c1, c2 = st.columns(2)
        model_key = c1.selectbox("Wybierz model prognozowania:", options=list(MODELS.keys()))
        selected_model = MODELS[model_key]
        
        forecast_periods = c2.number_input("Ile okresów w przyszłość prognozować?", min_value=1, value=12)
        
        seasonal_periods = None
        if selected_model.startswith('winters'):
            st.info("Model Wintersa wymaga zdefiniowania długości cyklu sezonowego.")
            seasonal_periods = st.number_input(
                "Okresy sezonowe (np. 12 dla danych miesięcznych, 7 dla dziennych, 4 dla kwartalnych)", 
                min_value=2, value=12
            )

        if st.button("🚀 Wygeneruj Prognozę", use_container_width=True, type="primary"):
            # <<< POCZĄTEK POPRAWIONEGO BLOKU OBSŁUGI BŁĘDÓW >>>
            try:
                with st.spinner("Buduję model i generuję prognozę..."):
                    model, forecast, metrics = run_forecasting_model(
                        st.session_state.ts_data, selected_model, forecast_periods, seasonal_periods
                    )
                    st.session_state.forecast_results = {"model": model, "forecast": forecast, "metrics": metrics}
                    st.session_state.analysis_generated = True
            
            except ValueError as ve:
                # Sprawdź, czy to jest błąd dotyczący sezonowości
                if 'less than two full seasonal cycles' in str(ve).lower():
                    needed = 2 * (seasonal_periods or 0)
                    available = len(st.session_state.ts_data)
                    st.error(
                        f"**Błąd: Za mało danych dla modelu sezonowego!**\n\n"
                        f"Wybrany Model Wintersa wymaga co najmniej dwóch pełnych cykli sezonowych "
                        f"({needed} okresów), a Twój zbiór danych ma tylko {available} okresów.\n\n"
                        f"**Co możesz zrobić?**\n"
                        f"1. Wybierz prostszy model, np. **Model Holta**.\n"
                        f"2. Wgraj plik z większą ilością danych historycznych."
                    )
                else:
                    # Inny błąd wartości, pokaż go ogólnie
                    st.error(f"Wystąpił błąd związany z danymi: {ve}")
                    st.code(traceback.format_exc())
            
            except Exception as e:
                # Pozostałe, nieoczekiwane błędy
                st.error(f"Wystąpił nieoczekiwany błąd podczas generowania prognozy: {e}")
                st.code(traceback.format_exc())
            # <<< KONIEC POPRAWIONEGO BLOKU OBSŁUGI BŁĘDÓW >>>


        if st.session_state.get('analysis_generated'):
            res = st.session_state.forecast_results
            history = st.session_state.ts_data
            forecast = res['forecast']
            
            st.markdown("---"); st.subheader("📊 Wyniki Prognozy")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history.index, y=history, mode='lines', name='Dane historyczne'))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Prognoza', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=res['model'].fittedvalues.index, y=res['model'].fittedvalues, mode='lines', name='Dopasowanie modelu', line=dict(color='rgba(255, 127, 14, 0.5)')))
            
            is_numeric_index = pd.api.types.is_numeric_dtype(history.index)
            axis_title = "Indeks numeryczny" if is_numeric_index else "Data"

            fig.update_layout(
                title=f"Prognoza na {forecast_periods} okresów (Model: {model_key})",
                xaxis_title=axis_title, yaxis_title="Wartość", legend_title="Legenda", plot_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Jakość dopasowania modelu")
                st.metric("Błąd Średni Bezwzględny (MAE)", f"{res['metrics']['MAE']:.3f}", help="Średnia bezwzględna różnica między wartościami rzeczywistymi a dopasowanymi przez model. Im niższa, tym lepiej.")
                st.metric("Pierwiastek Błędu Średniokwadratowego (RMSE)", f"{res['metrics']['RMSE']:.3f}", help="Mierzy odchylenie standardowe reszt modelu. Im niższy, tym lepiej.")
            
            with col2:
                st.subheader("Prognozowane wartości")
                forecast_df = pd.DataFrame({axis_title: forecast.index, 'Prognoza': forecast.values})
                st.dataframe(forecast_df.style.format({'Prognoza': '{:,.2f}'}), use_container_width=True)

    else:
        st.info("Najpierw przetwórz dane w Kroku 2, aby móc wygenerować prognozę.")

st.sidebar.title("O Aplikacji")
st.sidebar.info("Ta aplikacja wykorzystuje `statsmodels` i `plotly` do budowy i wizualizacji prognoz szeregów czasowych za pomocą modeli wygładzania wykładniczego.")
st.sidebar.markdown("---")
if st.sidebar.button("Zacznij od nowa (Reset)"):
    initialize_session_state()
    st.rerun()
