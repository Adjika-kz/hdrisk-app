import random
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="HDRisk",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Справочники ----------
SEX_MAP = {
    1: "Мужчина",
    0: "Женщина",
}

CP_MAP = {
    0: "0 — типичная стенокардия",
    1: "1 — атипичная стенокардия",
    2: "2 — неангинозная боль",
    3: "3 — бессимптомное течение",
}

RESTECG_MAP = {
    0: "0 — норма",
    1: "1 — нарушения ST-T",
    2: "2 — гипертрофия левого желудочка",
}

YES_NO_MAP = {
    0: "Нет",
    1: "Да",
}

SLOPE_MAP = {
    0: "0 — восходящий",
    1: "1 — ровный",
    2: "2 — нисходящий",
}

THAL_MAP = {
    0: "0 — не указано / неизвестно",
    1: "1 — нормальный",
    2: "2 — фиксированный дефект",
    3: "3 — обратимый дефект",
}


# ---------- Кэш модели ----------
@st.cache_resource
def load_model():
    return joblib.load("models/heart_model.joblib")


model = load_model()


# ---------- Session state ----------
DEFAULTS = {
    "age": 30,
    "sex": 1,
    "cp": 0,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def fill_random():
    st.session_state.age = random.randint(29, 77)
    st.session_state.sex = random.choice([1, 0])
    st.session_state.cp = random.choice([0, 1, 2, 3])
    st.session_state.trestbps = random.randint(90, 180)
    st.session_state.chol = random.randint(150, 350)
    st.session_state.fbs = random.choice([0, 1])
    st.session_state.restecg = random.choice([0, 1, 2])
    st.session_state.thalach = random.randint(90, 200)
    st.session_state.exang = random.choice([0, 1])
    st.session_state.oldpeak = round(random.uniform(0.0, 6.0), 1)
    st.session_state.slope = random.choice([0, 1, 2])
    st.session_state.ca = random.choice([0, 1, 2, 3, 4])
    st.session_state.thal = random.choice([0, 1, 2, 3])


def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# ---------- Шапка ----------
st.title("HDRisk — Прогноз риска сердечного заболевания")
st.markdown(
    "Оцените вероятность риска по медицинским показателям пациента. "
    "Приложение демонстрирует работу ML-модели на учебном датасете."
)

top_col1, top_col2, top_col3 = st.columns([1, 1, 2])

with top_col1:
    st.button("🎲 Случайный пример", on_click=fill_random, use_container_width=True)

with top_col2:
    st.button("↺ Сбросить", on_click=reset_defaults, use_container_width=True)

with top_col3:
    st.info("Сначала заполните форму, затем нажмите «Рассчитать риск».")


# ---------- Форма ----------
with st.form("risk_form", border=True):
    st.subheader("Данные пациента")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Основные параметры")
        age = st.number_input(
            "Возраст",
            min_value=1,
            max_value=120,
            step=1,
            key="age",
            help="Полный возраст пациента в годах.",
        )

        sex = st.radio(
            "Пол",
            options=[1, 0],
            key="sex",
            horizontal=True,
            format_func=lambda x: SEX_MAP[x],
            help="Биологический пол пациента.",
        )

        cp = st.selectbox(
            "Тип боли в груди",
            options=[0, 1, 2, 3],
            key="cp",
            format_func=lambda x: CP_MAP[x],
            help="Категория болевого синдрома в груди из медицинского датасета.",
        )

        trestbps = st.number_input(
            "Артериальное давление в покое, мм рт. ст.",
            min_value=80,
            max_value=220,
            step=1,
            key="trestbps",
            help="Систолическое давление после нескольких минут покоя.",
        )

        chol = st.number_input(
            "Общий холестерин, мг/дл",
            min_value=100,
            max_value=600,
            step=1,
            key="chol",
            help="Лабораторный показатель общего холестерина.",
        )

    with col2:
        st.markdown("#### Анализы и нагрузочный тест")
        fbs = st.radio(
            "Сахар крови натощак выше 120 мг/дл",
            options=[0, 1],
            key="fbs",
            horizontal=True,
            format_func=lambda x: YES_NO_MAP[x],
            help="Показатель измеряется натощак: 0 — нет, 1 — да.",
        )

        restecg = st.selectbox(
            "Результат ЭКГ в покое",
            options=[0, 1, 2],
            key="restecg",
            format_func=lambda x: RESTECG_MAP[x],
            help="Категория результата ЭКГ в покое.",
        )

        thalach = st.number_input(
            "Максимальный пульс",
            min_value=60,
            max_value=220,
            step=1,
            key="thalach",
            help="Максимальная достигнутая частота сердечных сокращений.",
        )

        exang = st.radio(
            "Стенокардия при физической нагрузке",
            options=[0, 1],
            key="exang",
            horizontal=True,
            format_func=lambda x: YES_NO_MAP[x],
            help="Возникает ли стенокардия при нагрузочном тесте.",
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            key="oldpeak",
            help="Снижение сегмента ST на фоне нагрузки по сравнению с покоем.",
        )

    col3, col4, col5 = st.columns(3)

    with col3:
        slope = st.selectbox(
            "Наклон сегмента ST",
            options=[0, 1, 2],
            key="slope",
            format_func=lambda x: SLOPE_MAP[x],
            help="Характер наклона сегмента ST на пике нагрузки.",
        )

    with col4:
        ca = st.selectbox(
            "Количество крупных сосудов",
            options=[0, 1, 2, 3, 4],
            key="ca",
            help="Количество крупных сосудов, выявленных при флюороскопии.",
        )

    with col5:
        thal = st.selectbox(
            "Thal",
            options=[0, 1, 2, 3],
            key="thal",
            format_func=lambda x: THAL_MAP[x],
            help="Категориальный показатель thal из исходного датасета.",
        )

    submitted = st.form_submit_button("Рассчитать риск", use_container_width=True)


# ---------- Предсказание ----------
if submitted:
    input_df = pd.DataFrame(
        [
            {
                "age": st.session_state.age,
                "sex": st.session_state.sex,
                "cp": st.session_state.cp,
                "trestbps": st.session_state.trestbps,
                "chol": st.session_state.chol,
                "fbs": st.session_state.fbs,
                "restecg": st.session_state.restecg,
                "thalach": st.session_state.thalach,
                "exang": st.session_state.exang,
                "oldpeak": st.session_state.oldpeak,
                "slope": st.session_state.slope,
                "ca": st.session_state.ca,
                "thal": st.session_state.thal,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    probability = float(model.predict_proba(input_df)[0][1])

    st.divider()
    st.subheader("Результат оценки")

    st.divider()
    st.subheader("Результат оценки")

    st.metric("Риск", f"{probability * 100:.1f}%")
    st.progress(probability)

    if prediction == 1:
        st.error("⚠️ Высокий риск сердечного заболевания")
    else:
        st.success("🟢 Низкий риск")

    if probability < 0.35:
        st.success("Низкий риск. Существенных факторов не выявлено.")
    elif probability < 0.65:
        st.warning("Средний риск. Рекомендуется дополнительное обследование.")
    else:
        st.error("Высокий риск. Рекомендуется консультация специалиста.")

    st.caption("0% — минимальный риск, 100% — максимальный")
   

    with st.expander("Посмотреть переданные модели данные"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.caption("Это демонстрационная ML-модель и не заменяет консультацию врача.")


# ---------- Справка ----------
with st.expander("Как понимать показатели"):
    st.markdown(
        """
**Возраст** — полный возраст пациента в годах.

**Артериальное давление в покое** — обычно измеряется после нескольких минут спокойного состояния.

**Общий холестерин** — показатель из анализа крови.

**Сахар крови натощак выше 120 мг/дл** — бинарный признак: ниже/выше порога.

**Максимальный пульс** — максимальная ЧСС, обычно в рамках нагрузочного теста.

**ЭКГ, Oldpeak, наклон ST, количество сосудов, Thal** — специализированные медицинские показатели, которые в учебном датасете представлены кодами.
"""
    )