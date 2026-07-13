
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Steel DPP Carbon Predictor",
    page_icon="",
    layout="wide"
)


# =========================================================
# 1. GENERATE MOCK STEEL DPP DATA
# =========================================================
@st.cache_data
def generate_mock_data(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    product_types = np.array([
        "Hot Rolled Coil",
        "Wire Rod",
        "Galvanised Coil",
        "Electrical Steel",
        "Stainless Steel"
    ])

    steel_grades = np.array([
        "S235JR",
        "S355JR",
        "DX51D",
        "304",
        "M270-35A"
    ])

    routes = np.array(["EAF", "BF-BOF"])
    countries = np.array([
        "Germany",
        "France",
        "Italy",
        "Spain",
        "Belgium"
    ])

    product_type = rng.choice(
        product_types,
        n,
        p=[0.30, 0.18, 0.22, 0.15, 0.15]
    )

    steel_grade = rng.choice(steel_grades, n)
    technology_route = rng.choice(routes, n, p=[0.55, 0.45])
    country_of_origin = rng.choice(countries, n)

    recycled_content_pct = np.where(
        technology_route == "EAF",
        rng.normal(78, 10, n),
        rng.normal(24, 9, n)
    )
    recycled_content_pct = np.clip(recycled_content_pct, 5, 98)

    renewable_energy_pct = rng.uniform(5, 95, n)

    energy_consumption_kwh_per_tonne = np.where(
        technology_route == "EAF",
        rng.normal(650, 85, n),
        rng.normal(1850, 170, n)
    )
    energy_consumption_kwh_per_tonne = np.clip(
        energy_consumption_kwh_per_tonne,
        400,
        2300
    )

    scrap_quality_score = rng.uniform(55, 98, n)
    batch_weight_tonnes = rng.uniform(8, 35, n)
    thickness_mm = rng.uniform(0.5, 25, n)

    carbon_pct = np.clip(
        rng.normal(0.18, 0.07, n),
        0.02,
        0.65
    )

    manganese_pct = np.clip(
        rng.normal(1.15, 0.35, n),
        0.25,
        2.20
    )

    chromium_pct = np.where(
        product_type == "Stainless Steel",
        rng.normal(18.0, 1.2, n),
        np.clip(rng.normal(0.35, 0.20, n), 0, 1.5)
    )
    chromium_pct = np.clip(chromium_pct, 0, 22)

    yield_strength_mpa = (
        180
        + 130 * carbon_pct
        + 45 * manganese_pct
        + 3.5 * chromium_pct
        + rng.normal(0, 25, n)
    )
    yield_strength_mpa = np.clip(
        yield_strength_mpa,
        170,
        750
    )

    tensile_strength_mpa = (
        yield_strength_mpa
        + rng.normal(150, 30, n)
    )
    tensile_strength_mpa = np.clip(
        tensile_strength_mpa,
        300,
        950
    )

    route_effect = np.where(
        technology_route == "BF-BOF",
        520,
        90
    )

    product_effect_map = {
        "Hot Rolled Coil": 40,
        "Wire Rod": 55,
        "Galvanised Coil": 130,
        "Electrical Steel": 170,
        "Stainless Steel": 360
    }

    product_effect = np.array([
        product_effect_map[item]
        for item in product_type
    ])

    # Mock carbon-footprint relationship.
    # Noise is intentionally low so the model learns the direction clearly.
    carbon_footprint = (
        0.75 * energy_consumption_kwh_per_tonne
        + route_effect
        + product_effect
        - 4.5 * recycled_content_pct
        - 2.5 * renewable_energy_pct
        - 1.2 * scrap_quality_score
        + 0.8 * chromium_pct
        + rng.normal(0, 25, n)
    )

    carbon_footprint = np.clip(
        carbon_footprint,
        120,
        2800
    )

    df = pd.DataFrame({
        "batch_id": [
            f"BATCH-{i:05d}"
            for i in range(1, n + 1)
        ],
        "product_type": product_type,
        "steel_grade": steel_grade,
        "technology_route": technology_route,
        "country_of_origin": country_of_origin,
        "batch_weight_tonnes": np.round(batch_weight_tonnes, 2),
        "thickness_mm": np.round(thickness_mm, 2),
        "recycled_content_pct": np.round(recycled_content_pct, 2),
        "renewable_energy_pct": np.round(renewable_energy_pct, 2),
        "energy_consumption_kwh_per_tonne": np.round(
            energy_consumption_kwh_per_tonne,
            1
        ),
        "scrap_quality_score": np.round(scrap_quality_score, 2),
        "carbon_pct": np.round(carbon_pct, 3),
        "manganese_pct": np.round(manganese_pct, 3),
        "chromium_pct": np.round(chromium_pct, 3),
        "yield_strength_mpa": np.round(yield_strength_mpa, 1),
        "tensile_strength_mpa": np.round(tensile_strength_mpa, 1),
        "carbon_footprint_kgco2e_per_tonne": np.round(
            carbon_footprint,
            1
        )
    })

    return df


# =========================================================
# 2. TRAIN A MONOTONIC REGRESSION MODEL
# =========================================================
@st.cache_resource
def train_model(df):
    target = "carbon_footprint_kgco2e_per_tonne"

    features = [
        "product_type",
        "steel_grade",
        "technology_route",
        "country_of_origin",
        "batch_weight_tonnes",
        "thickness_mm",
        "recycled_content_pct",
        "renewable_energy_pct",
        "energy_consumption_kwh_per_tonne",
        "scrap_quality_score",
        "carbon_pct",
        "manganese_pct",
        "chromium_pct",
        "yield_strength_mpa",
        "tensile_strength_mpa"
    ]

    X = df[features].copy()
    y = df[target].astype(float).copy()

    categorical_features = [
        "product_type",
        "steel_grade",
        "technology_route",
        "country_of_origin"
    ]

    numeric_features = [
        col for col in features
        if col not in categorical_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Monotonic constraints:
    # +1 means prediction must increase when the feature increases
    # -1 means prediction must decrease when the feature increases
    #  0 means no forced direction
    constraints = []

    for name in feature_names:
        if "energy_consumption_kwh_per_tonne" in name:
            constraints.append(1)
        elif "recycled_content_pct" in name:
            constraints.append(-1)
        elif "renewable_energy_pct" in name:
            constraints.append(-1)
        elif "scrap_quality_score" in name:
            constraints.append(-1)
        else:
            constraints.append(0)

    model = HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        monotonic_cst=constraints,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "features": features,
        "feature_names": feature_names
    }


def predict_footprint(model_bundle, input_df):
    processed = model_bundle["preprocessor"].transform(input_df)
    prediction = model_bundle["model"].predict(processed)
    return prediction


# =========================================================
# 3. LOAD DATA AND MODEL
# =========================================================
df = generate_mock_data()
model_bundle = train_model(df)
metrics = model_bundle["metrics"]


# =========================================================
# 4. USER INTERFACE
# =========================================================
# =========================================================
# 4. SPHERITY-INSPIRED USER INTERFACE
# =========================================================
SPHERITY_GREEN = "#27E5A8"
SPHERITY_DARK = "#12231F"
SPHERITY_SOFT_GREEN = "#EAFBF5"
SPHERITY_BORDER = "#D7E9E2"
SPHERITY_MUTED = "#60716B"
SPHERITY_WHITE = "#FFFFFF"

st.markdown(
    f"""
    <style>
        .stApp {{
            background: linear-gradient(180deg, {SPHERITY_WHITE} 0%, #F7FCFA 100%);
            color: {SPHERITY_DARK};
        }}

        [data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.92);
            border-bottom: 1px solid {SPHERITY_BORDER};
        }}

        [data-testid="stSidebar"] {{
            background: {SPHERITY_SOFT_GREEN};
            border-right: 1px solid {SPHERITY_BORDER};
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.25rem;
        }}

        .block-container {{
            max-width: 1280px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, h4 {{
            color: {SPHERITY_DARK};
            letter-spacing: -0.02em;
        }}

        h1 {{
            font-size: 2.35rem !important;
            font-weight: 750 !important;
        }}

        h2 {{
            font-size: 1.65rem !important;
            font-weight: 700 !important;
        }}

        h3 {{
            font-size: 1.15rem !important;
            font-weight: 700 !important;
        }}

        .spherity-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            background: {SPHERITY_WHITE};
            border: 1px solid {SPHERITY_BORDER};
            border-radius: 20px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 30px rgba(18, 35, 31, 0.06);
        }}

        .spherity-brand {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }}

        .spherity-logo-image {{
            width: 52px;
            height: 52px;
            object-fit: contain;
            display: block;
        }}

        .spherity-sidebar-logo {{
            width: 42px;
            height: 42px;
            object-fit: contain;
            display: block;
            margin-bottom: 0.55rem;
        }}

        .spherity-wordmark {{
            font-size: 1.55rem;
            line-height: 1;
            font-weight: 800;
            color: {SPHERITY_DARK};
        }}

        .spherity-product {{
            margin-top: 0.28rem;
            font-size: 0.82rem;
            color: {SPHERITY_MUTED};
            letter-spacing: 0.02em;
        }}

        .demo-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            background: {SPHERITY_SOFT_GREEN};
            border: 1px solid {SPHERITY_BORDER};
            color: {SPHERITY_DARK};
            font-size: 0.8rem;
            font-weight: 650;
            white-space: nowrap;
        }}

        .hero-panel {{
            border-radius: 22px;
            padding: 1.7rem 1.8rem;
            margin: 0 0 1.4rem 0;
            background: linear-gradient(135deg, {SPHERITY_SOFT_GREEN} 0%, #FFFFFF 75%);
            border: 1px solid {SPHERITY_BORDER};
        }}

        .hero-kicker {{
            display: inline-block;
            color: #0B7E5A;
            background: rgba(39, 229, 168, 0.16);
            border-radius: 999px;
            padding: 0.38rem 0.72rem;
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.8rem;
        }}

        .hero-title {{
            color: {SPHERITY_DARK};
            font-size: 2.15rem;
            line-height: 1.12;
            font-weight: 780;
            margin-bottom: 0.6rem;
        }}

        .hero-copy {{
            color: {SPHERITY_MUTED};
            font-size: 1rem;
            line-height: 1.65;
            max-width: 850px;
            margin: 0;
        }}

        .section-label {{
            font-size: 0.76rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #0B7E5A;
            margin-bottom: 0.25rem;
        }}

        .scenario-card {{
            background: {SPHERITY_WHITE};
            border: 1px solid {SPHERITY_BORDER};
            border-radius: 18px;
            padding: 1rem 1.1rem 0.3rem 1.1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 6px 20px rgba(18, 35, 31, 0.04);
        }}

        [data-testid="stMetric"] {{
            background: {SPHERITY_WHITE};
            border: 1px solid {SPHERITY_BORDER};
            border-radius: 17px;
            padding: 1rem 1.05rem;
            box-shadow: 0 6px 20px rgba(18, 35, 31, 0.045);
        }}

        [data-testid="stMetricLabel"] {{
            color: {SPHERITY_MUTED};
            font-weight: 600;
        }}

        [data-testid="stMetricValue"] {{
            color: {SPHERITY_DARK};
            font-weight: 760;
        }}

        [data-testid="stMetricDelta"] svg {{
            fill: #0B7E5A;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        .stNumberInput input {{
            background: {SPHERITY_WHITE} !important;
            border-color: {SPHERITY_BORDER} !important;
            border-radius: 10px !important;
        }}

        .stSlider [data-baseweb="slider"] div[role="slider"] {{
            background-color: {SPHERITY_GREEN};
            border-color: {SPHERITY_GREEN};
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background: {SPHERITY_GREEN};
            color: {SPHERITY_DARK};
            border: none;
            border-radius: 10px;
            font-weight: 750;
            padding: 0.55rem 1rem;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: #1FD39A;
            color: {SPHERITY_DARK};
            border: none;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {SPHERITY_BORDER};
            border-radius: 14px;
            overflow: hidden;
        }}

        .info-card {{
            background: {SPHERITY_WHITE};
            border: 1px solid {SPHERITY_BORDER};
            border-left: 4px solid {SPHERITY_GREEN};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin: 0.8rem 0 1.1rem 0;
            color: {SPHERITY_MUTED};
            line-height: 1.6;
        }}

        .model-note {{
            background: {SPHERITY_SOFT_GREEN};
            border: 1px solid {SPHERITY_BORDER};
            border-radius: 14px;
            padding: 1rem 1.15rem;
            margin-top: 1rem;
        }}

        hr {{
            border-color: {SPHERITY_BORDER} !important;
        }}

        footer {{
            visibility: hidden;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


def render_brand_header():
    st.markdown(
        """
        <div class="spherity-header">
            <div class="spherity-brand">
                <img
                    class="spherity-logo-image"
                    src="https://www.spherity.com/favicon.ico"
                    alt="Spherity logo"
                />
                <div>
                    <div class="spherity-wordmark">SPHERITY</div>
                    <div class="spherity-product">Steel Digital Product Passport Analytics</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


render_brand_header()

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.3rem 0 0.8rem 0;">
            <img
                class="spherity-sidebar-logo"
                src="https://www.spherity.com/favicon.ico"
                alt="Spherity logo"
            />
            <div style="font-size:1.25rem;font-weight:800;color:#12231F;">SPHERITY</div>
            <div style="font-size:0.78rem;color:#60716B;margin-top:0.2rem;">Steel DPP Demo</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Navigation")

    page = st.radio(
        "Choose a workspace",
        [
            "Scenario simulator",
            "Mock data explorer",
            "Model performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption(
        "This prototype uses synthetic data and a deliberately constrained "
        "regression model for demonstration purposes only."
    )


# =========================================================
# PAGE 1: SCENARIO SIMULATOR
# =========================================================
if page == "Scenario simulator":
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Steel DPP scenario intelligence</div>
            <div class="hero-title">Explore potential carbon-footprint improvements</div>
            <p class="hero-copy">
                Compare a current steel production configuration with an improved scenario.
                The model estimates the carbon footprint for each configuration in kg CO₂e
                per tonne and shows the expected difference.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-label">Scenario inputs</div>', unsafe_allow_html=True)
    st.subheader("Configure the two production scenarios")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
        st.markdown("### Current process")
        st.caption("Baseline production and material parameters")

        product_type = st.selectbox(
            "Product type",
            sorted(df["product_type"].unique()),
            index=2
        )

        steel_grade = st.selectbox(
            "Steel grade",
            sorted(df["steel_grade"].unique()),
            index=3
        )

        technology_route = st.selectbox(
            "Technology route",
            ["EAF", "BF-BOF"],
            index=0,
            help="EAF uses an electric arc furnace. BF-BOF uses the blast furnace/basic oxygen furnace route."
        )

        country_of_origin = st.selectbox(
            "Country of origin",
            sorted(df["country_of_origin"].unique()),
            index=2
        )

        batch_weight_tonnes = st.slider(
            "Batch weight (tonnes)", 8.0, 35.0, 25.0, 0.5
        )

        thickness_mm = st.slider(
            "Thickness (mm)", 0.5, 25.0, 4.0, 0.5
        )

        recycled_content_pct = st.slider(
            "Current recycled content (%)", 5.0, 98.0, 65.0, 1.0
        )

        renewable_energy_pct = st.slider(
            "Current renewable energy (%)", 0.0, 100.0, 30.0, 1.0
        )

        energy_consumption = st.slider(
            "Current energy consumption (kWh/t)", 400.0, 2300.0, 720.0, 10.0
        )

        scrap_quality_score = st.slider(
            "Current scrap quality score", 50.0, 100.0, 80.0, 1.0,
            help="A higher value represents cleaner and more consistent scrap input in this synthetic dataset."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
        st.markdown("### Improved process")
        st.caption("Potential sustainability improvements and material properties")

        improved_recycled = st.slider(
            "Improved recycled content (%)", 5.0, 98.0, 85.0, 1.0
        )

        improved_renewable = st.slider(
            "Improved renewable energy (%)", 0.0, 100.0, 70.0, 1.0
        )

        improved_energy = st.slider(
            "Improved energy consumption (kWh/t)", 400.0, 2300.0, 610.0, 10.0
        )

        improved_scrap_quality = st.slider(
            "Improved scrap quality score", 50.0, 100.0, 92.0, 1.0
        )

        carbon_pct = st.number_input(
            "Carbon (%)", min_value=0.02, max_value=0.65, value=0.18, step=0.01
        )

        manganese_pct = st.number_input(
            "Manganese (%)", min_value=0.25, max_value=2.20, value=1.25, step=0.05
        )

        chromium_pct = st.number_input(
            "Chromium (%)", min_value=0.0, max_value=22.0, value=0.25, step=0.05
        )

        yield_strength_mpa = st.number_input(
            "Yield strength (MPa)", min_value=170.0, max_value=750.0, value=355.0, step=5.0
        )

        tensile_strength_mpa = st.number_input(
            "Tensile strength (MPa)", min_value=300.0, max_value=950.0, value=510.0, step=5.0
        )
        st.markdown('</div>', unsafe_allow_html=True)

    current = pd.DataFrame([{
        "product_type": product_type,
        "steel_grade": steel_grade,
        "technology_route": technology_route,
        "country_of_origin": country_of_origin,
        "batch_weight_tonnes": batch_weight_tonnes,
        "thickness_mm": thickness_mm,
        "recycled_content_pct": recycled_content_pct,
        "renewable_energy_pct": renewable_energy_pct,
        "energy_consumption_kwh_per_tonne": energy_consumption,
        "scrap_quality_score": scrap_quality_score,
        "carbon_pct": carbon_pct,
        "manganese_pct": manganese_pct,
        "chromium_pct": chromium_pct,
        "yield_strength_mpa": yield_strength_mpa,
        "tensile_strength_mpa": tensile_strength_mpa
    }])

    improved = current.copy()
    improved["recycled_content_pct"] = improved_recycled
    improved["renewable_energy_pct"] = improved_renewable
    improved["energy_consumption_kwh_per_tonne"] = improved_energy
    improved["scrap_quality_score"] = improved_scrap_quality

    current_prediction = predict_footprint(model_bundle, current)[0]
    improved_prediction = predict_footprint(model_bundle, improved)[0]

    reduction = current_prediction - improved_prediction
    reduction_pct = reduction / current_prediction * 100 if current_prediction > 0 else 0

    st.divider()
    st.markdown('<div class="section-label">Prediction summary</div>', unsafe_allow_html=True)
    st.subheader("Estimated scenario impact")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Current predicted footprint",
        f"{current_prediction:,.0f} kg CO₂e/t"
    )
    col2.metric(
        "Improved predicted footprint",
        f"{improved_prediction:,.0f} kg CO₂e/t"
    )
    col3.metric(
        "Estimated reduction",
        f"{reduction:,.0f} kg CO₂e/t",
        f"{reduction_pct:.1f}%"
    )

    chart_df = pd.DataFrame({
        "Scenario": ["Current process", "Improved process"],
        "Predicted kg CO₂e/t": [current_prediction, improved_prediction]
    }).set_index("Scenario")

    st.bar_chart(chart_df, color=SPHERITY_GREEN)

    if reduction > 0:
        st.success(
            f"The improved scenario is predicted to reduce the carbon footprint "
            f"by approximately {reduction:,.0f} kg CO₂e/t, or {reduction_pct:.1f}%."
        )
    elif reduction < 0:
        st.warning(
            f"The improved scenario is predicted to increase the carbon footprint "
            f"by approximately {abs(reduction):,.0f} kg CO₂e/t, or {abs(reduction_pct):.1f}%."
        )
    else:
        st.info("The model predicts no meaningful change between the scenarios.")

    st.markdown("### Input changes")
    st.caption("Only these four sustainability-related inputs differ between the scenarios.")

    change_table = pd.DataFrame({
        "Variable": [
            "Recycled content (%)",
            "Renewable energy (%)",
            "Energy consumption (kWh/t)",
            "Scrap quality score"
        ],
        "Current": [
            recycled_content_pct,
            renewable_energy_pct,
            energy_consumption,
            scrap_quality_score
        ],
        "Improved": [
            improved_recycled,
            improved_renewable,
            improved_energy,
            improved_scrap_quality
        ]
    })

    change_table["Change"] = change_table["Improved"] - change_table["Current"]

    st.dataframe(
        change_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        <div class="info-card">
            <strong>How to read this result:</strong> the chart compares the predicted
            footprint of one tonne of steel under the two input configurations. A lower
            improved bar indicates a potentially better carbon outcome. This is a model
            estimate based on synthetic data, not a verified environmental product declaration.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 2: MOCK DATA EXPLORER
# =========================================================
elif page == "Mock data explorer":
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Digital Product Passport data</div>
            <div class="hero-title">Explore the synthetic Steel DPP backend</div>
            <p class="hero-copy">
                Filter the demonstration dataset by steel product and production route,
                review summary indicators, and export the selected records.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_products = st.multiselect(
            "Product types",
            sorted(df["product_type"].unique()),
            default=sorted(df["product_type"].unique())
        )

    with filter_col2:
        selected_routes = st.multiselect(
            "Technology routes",
            sorted(df["technology_route"].unique()),
            default=sorted(df["technology_route"].unique())
        )

    filtered = df[
        df["product_type"].isin(selected_products)
        & df["technology_route"].isin(selected_routes)
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Batches", f"{len(filtered):,}")

    avg_footprint = filtered["carbon_footprint_kgco2e_per_tonne"].mean()
    avg_recycled = filtered["recycled_content_pct"].mean()

    col2.metric(
        "Average carbon footprint",
        f"{avg_footprint:,.0f} kg CO₂e/t" if not np.isnan(avg_footprint) else "No data"
    )
    col3.metric(
        "Average recycled content",
        f"{avg_recycled:.1f}%" if not np.isnan(avg_recycled) else "No data"
    )

    st.markdown("### Filtered Steel DPP records")
    st.dataframe(filtered, use_container_width=True, height=500)

    st.download_button(
        "Download filtered data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="spherity_steel_dpp_demo_data.csv",
        mime="text/csv"
    )


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================
elif page == "Model performance":
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Model transparency</div>
            <div class="hero-title">Understand the prediction model</div>
            <p class="hero-copy">
                Review the test-set performance and the reasoning behind the constrained
                gradient-boosting model used in this demonstration.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['MAE']:.1f} kg CO₂e/t")
    col2.metric("RMSE", f"{metrics['RMSE']:.1f} kg CO₂e/t")
    col3.metric("R²", f"{metrics['R2']:.3f}")

    st.markdown("### Model used")
    st.markdown(
        """
        <div class="model-note">
            <strong>Histogram-based Gradient Boosting Regressor</strong><br><br>
            The model combines many small decision trees. Each new tree focuses on correcting
            errors made by the previous trees. This makes gradient boosting suitable for
            structured DPP data containing non-linear relationships and interactions between
            production, material, and sustainability variables.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Why this model was chosen")
    st.write(
        "It can model complex relationships without requiring a single fixed equation, "
        "and it works well with mixed numerical and categorical variables after preprocessing."
    )

    st.markdown("### What the values mean")
    metric_help = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Meaning": [
            "The average absolute difference between the predicted and generated footprint values.",
            "An error measure that penalizes large prediction mistakes more strongly than MAE.",
            "The share of variation in the synthetic target explained by the model. Values closer to 1 are better."
        ],
        "Displayed value": [
            f"{metrics['MAE']:.1f} kg CO₂e/t",
            f"{metrics['RMSE']:.1f} kg CO₂e/t",
            f"{metrics['R2']:.3f}"
        ]
    })
    st.dataframe(metric_help, use_container_width=True, hide_index=True)

    st.warning(
        "Important: the dataset and target relationship are synthetic and intentionally "
        "simplified. The strong performance does not prove that the model would achieve "
        "the same accuracy with real steel-production or verified DPP data."
    )

st.markdown(
    """
    <div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #D7E9E2;
                color:#60716B;font-size:0.78rem;text-align:center;">
        Spherity Steel DPP analytics prototype · Synthetic data · Demonstration only
    </div>
    """,
    unsafe_allow_html=True
)
