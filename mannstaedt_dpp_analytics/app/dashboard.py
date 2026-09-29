from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.services.data_service import (
    get_available_datasets,
    load_dataset,
    load_mock_events,
    load_mock_sessions,
    load_mock_traceability,
    prepare_engagement_ml_data,
    prepare_section_performance_data,
    prepare_supplier_risk_data,
    prepare_traceability_ml_data,
)

from app.services.data_preparation_service import (
    prepare_dataset_for_ml,
)

from app.services.engine_service import (
    MODEL_OPTIONS,
    FUNCTION_OPTIONS,
    run_engine,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=
        "Mannstaedt DPP Analytics",

    page_icon=
        "📊",

    layout=
        "wide",

    initial_sidebar_state=
        "expanded",
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


LOGO_PATH = (
    PROJECT_ROOT
    / "app"
    / "assets"
    / "gmh_gruppe-logo.svg"
)


# =========================================================
# COLORS
# =========================================================

GMH_RED = "#E30613"

CARD_BACKGROUND = "#18191D"

CARD_BORDER = "#292B30"

SECONDARY_TEXT = "#92959D"


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background: #101114;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: #0C0D0F;
    border-right: 1px solid #23252A;
}

h1,
h2,
h3 {
    color: #F4F4F5 !important;
}

p {
    color: #B1B3B8;
}


/* =====================================================
   METRICS
===================================================== */

[data-testid="stMetric"] {

    background:
        #18191D;

    border:
        1px solid
        #292B30;

    border-radius:
        10px;

    padding:
        16px 18px;
}


[data-testid="stMetricLabel"] {

    color:
        #8D9097;

    font-size:
        11px;

    text-transform:
        uppercase;

    letter-spacing:
        .04em;
}


[data-testid="stMetricValue"] {

    color:
        white;

    font-size:
        28px;

    font-weight:
        700;
}


/* =====================================================
   SELECT
===================================================== */

div[data-baseweb="select"] > div {

    background:
        #18191D !important;

    border-color:
        #303238 !important;

    color:
        white !important;
}


div[data-baseweb="select"] span {

    color:
        white !important;
}


/* =====================================================
   MULTISELECT
===================================================== */

span[data-baseweb="tag"] {

    background:
        #E30613 !important;

    color:
        white !important;
}


/* =====================================================
   BUTTON
===================================================== */

.stButton > button {

    background:
        #E30613;

    color:
        white;

    border:
        none;

    border-radius:
        8px;

    min-height:
        45px;

    font-weight:
        650;
}


.stButton > button:hover {

    background:
        #B90510;

    color:
        white;
}


/* =====================================================
   EXPANDER
===================================================== */

[data-testid="stExpander"] {

    background:
        #16171A;

    border:
        1px solid
        #292B30;

    border-radius:
        9px;
}


/* =====================================================
   TABLE
===================================================== */

[data-testid="stDataFrame"] {

    border:
        1px solid
        #292B30;

    border-radius:
        9px;
}


/* =====================================================
   HEADER
===================================================== */

.mannstaedt-title {

    color:
        white;

    font-size:
        31px;

    font-weight:
        720;
}


.mannstaedt-subtitle {

    color:
        #8F929A;

    margin-top:
        5px;

    margin-bottom:
        30px;
}


/* =====================================================
   SECTION
===================================================== */

.section-kicker {

    color:
        #E30613;

    font-size:
        11px;

    font-weight:
        700;

    letter-spacing:
        .08em;

    text-transform:
        uppercase;
}


/* =====================================================
   PROTOTYPE
===================================================== */

.prototype-badge {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        7px;

    padding:
        6px 11px;

    background:
        rgba(
            227,
            6,
            19,
            .08
        );

    border:
        1px solid
        rgba(
            227,
            6,
            19,
            .32
        );

    border-radius:
        100px;

    color:
        #D1D2D5;

    font-size:
        12px;

    margin-bottom:
        22px;
}


.prototype-dot {

    width:
        7px;

    height:
        7px;

    background:
        #E30613;

    border-radius:
        50%;
}


/* =====================================================
   INSIGHT
===================================================== */

.insight-card {

    background:
        #18191D;

    border:
        1px solid
        #292B30;

    border-left:
        3px solid
        #E30613;

    border-radius:
        8px;

    padding:
        18px;

    margin:
        15px 0;

    color:
        #CACBCF;
}


.insight-card b {

    color:
        white;
}


/* =====================================================
   MODE CARD
===================================================== */

.mode-card {

    background:
        #15161A;

    border:
        1px solid
        #292B30;

    border-radius:
        10px;

    padding:
        18px;

    margin:
        15px 0 22px 0;
}


.mode-title {

    color:
        white;

    font-size:
        16px;

    font-weight:
        650;
}


.mode-description {

    color:
        #9699A1;

    font-size:
        13px;

    margin-top:
        5px;
}


/* =====================================================
   PREPARATION BOX
===================================================== */

.preparation-box {

    background:
        #15171A;

    border:
        1px solid
        #303238;

    border-left:
        3px solid
        #E30613;

    border-radius:
        10px;

    padding:
        18px;

    margin:
        15px 0 22px 0;
}


.preparation-title {

    color:
        #FFFFFF;

    font-size:
        15px;

    font-weight:
        650;
}


.preparation-subtitle {

    color:
        #90939A;

    font-size:
        12px;

    margin-top:
        4px;
}


/* =====================================================
   MODEL BADGE
===================================================== */

.model-badge {

    display:
        inline-block;

    padding:
        4px 9px;

    border-radius:
        20px;

    background:
        rgba(
            255,
            255,
            255,
            .05
        );

    border:
        1px solid
        #303238;

    color:
        #B8BAC0;

    font-size:
        11px;

    margin-bottom:
        15px;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HUMAN LABELS
# =========================================================

FEATURE_NAMES = {

    "qr_entry":
        "Entered via QR",

    "visit_duration":
        "Visit Duration",

    "sections_viewed":
        "Sections Viewed",

    "documents_opened":
        "Documents Opened",

    "opened_document":
        "Opened a Document",

    "risk_score":
        "Traceability Risk Score",

    "high_risk":
        "High Traceability Risk",

    "missing_link":
        "Missing Traceability Link",
}


MODEL_FRIENDLY_NAMES = {

    "linear-regression":
        "Linear Regression",

    "logistic-regression":
        "Logistic Regression",

    "random-forest-regression":
        "Random Forest Regressor",

    "random-forest-classification":
        "Random Forest Classifier",

    "cluster":
        "K-Means Clustering",

    "isolation-forest":
        "Isolation Forest",

    "feature-selection":
        "Feature Selection",
}


FUNCTION_FRIENDLY_NAMES = {

    "predict":
        "Train & Evaluate",

    "result":
        "Actual vs Predicted",

    "confusion":
        "Confusion Matrix",

    "predict_input":
        "Predict Scenario",

    "elbow":
        "Elbow Analysis",

    "silhouette":
        "Silhouette Analysis",

    "correlation":
        "Correlation Analysis",

    "correlate":
        "Highly Correlated Features",

    "correlated":
        "Remove Correlated Features",

    "pca_info":
        "PCA Information",

    "anova_regression":
        "ANOVA Regression",

    "mutual_info_regression":
        "Mutual Information",

    "lasso":
        "Lasso Feature Selection",
}


def friendly_feature(
    name,
):

    return FEATURE_NAMES.get(
        name,
        name.replace(
            "_",
            " ",
        ).title(),
    )


# =========================================================
# BASIC UI HELPERS
# =========================================================

def show_header():

    st.markdown(
        """
<div class="mannstaedt-title">
Mannstaedt DPP Analytics
</div>

<div class="mannstaedt-subtitle">
Digital Product Passport Intelligence
</div>
""",
        unsafe_allow_html=True,
    )


def prototype_badge():

    st.markdown(
        """
<div class="prototype-badge">
<span class="prototype-dot"></span>
Synthetic prototype data
</div>
""",
        unsafe_allow_html=True,
    )


def insight(
    title,
    text,
):

    st.markdown(
        f"""
<div class="insight-card">
<b>{title}</b><br>
{text}
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# DATA PREPARATION SUMMARY
# =========================================================

def show_data_preparation_summary(
    summary,
):

    st.markdown(
        """
<div class="preparation-box">

<div class="preparation-title">
Data Preparation Summary
</div>

<div class="preparation-subtitle">
What happened automatically before the dataset
became available to the machine-learning engine.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(
        "Rows Loaded",
        f"{summary['original_rows']:,}",
    )


    c2.metric(
        "Rows Ready for ML",
        f"{summary['final_rows']:,}",
    )


    c3.metric(
        "Rows Removed",
        f"{summary['rows_removed']:,}",
    )


    c4.metric(
        "Model-Ready Columns",
        summary[
            "final_columns"
        ],
    )


    with st.expander(
        "What changed during preparation?"
    ):

        st.write(
            "### Row cleaning"
        )


        st.write(
            "Duplicate rows removed:",
            summary[
                "duplicates_removed"
            ],
        )


        st.write(
            "Infinite numeric values cleaned:",
            summary[
                "infinite_values_cleaned"
            ],
        )


        st.write(
            "Missing numeric values filled:",
            summary[
                "missing_values_filled"
            ],
        )


        if (
            summary[
                "columns_with_missing"
            ]
        ):

            st.write(
                "Columns containing missing values:"
            )


            st.write(
                [
                    friendly_feature(
                        column
                    )

                    for column
                    in summary[
                        "columns_with_missing"
                    ]
                ]
            )


        st.divider()


        st.write(
            "### Column preparation"
        )


        encoded_columns = (
            summary[
                "encoded_columns"
            ]
        )


        if encoded_columns:

            st.write(
                "Categorical columns encoded:"
            )


            st.write(
                [
                    friendly_feature(
                        column
                    )

                    for column
                    in encoded_columns
                ]
            )


        else:

            st.write(
                "Categorical columns encoded: None"
            )


        excluded_columns = (
            summary[
                "excluded_columns"
            ]
        )


        if excluded_columns:

            st.write(
                "Columns excluded from ML:"
            )


            st.write(
                [
                    friendly_feature(
                        column
                    )

                    for column
                    in excluded_columns
                ]
            )


        else:

            st.write(
                "Columns excluded from ML: None"
            )


        st.divider()


        st.write(
            "### Final ML columns"
        )


        st.write(
            [
                friendly_feature(
                    column
                )

                for column
                in summary[
                    "model_columns"
                ]
            ]
        )


# =========================================================
# DARK BAR CHART
# =========================================================

def dark_bar(
    df,
    x,
    y,
    y_title,
    height=300,
):

    if df.empty:

        st.info(
            "No data available."
        )

        return


    chart = (
        alt.Chart(
            df
        )

        .mark_bar(
            color=
                GMH_RED,

            cornerRadiusTopLeft=
                4,

            cornerRadiusTopRight=
                4,
        )

        .encode(

            x=alt.X(
                x,
                title=None,
                axis=alt.Axis(
                    labelColor=
                        SECONDARY_TEXT,

                    grid=False,
                ),
            ),

            y=alt.Y(
                y,
                title=
                    y_title,

                axis=alt.Axis(
                    labelColor=
                        SECONDARY_TEXT,

                    titleColor=
                        SECONDARY_TEXT,

                    gridColor=
                        CARD_BORDER,
                ),
            ),

            tooltip=[
                x,
                y,
            ],
        )

        .properties(
            height=
                height,
        )

        .configure_view(
            fill=
                CARD_BACKGROUND,

            strokeOpacity=
                0,
        )

        .configure(
            background=
                CARD_BACKGROUND,
        )
    )


    st.altair_chart(
        chart,
        use_container_width=True,
        theme=None,
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def importance_chart(
    importance,
):

    if not importance:

        return


    df = pd.DataFrame(
        {
            "Feature":
                list(
                    importance.keys()
                ),

            "Importance":
                list(
                    importance.values()
                ),
        }
    )


    df[
        "Feature"
    ] = [
        friendly_feature(
            value
        )

        for value
        in df[
            "Feature"
        ]
    ]


    df = (
        df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .head(12)
    )


    dark_bar(
        df,
        "Feature:N",
        "Importance:Q",
        "Relative Importance",
        height=330,
    )


# =========================================================
# ADVANCED RESULT DISPLAY
# =========================================================

def display_advanced_result(
    result,
    algorithm_name,
    function_name,
):

    # -----------------------------------------------------
    # RANDOM FOREST CLASSIFIER
    # -----------------------------------------------------

    if (
        algorithm_name
        == "random-forest-classification"
        and isinstance(
            result,
            dict,
        )
        and function_name
        == "predict"
    ):

        c1, c2, c3, c4 = (
            st.columns(4)
        )


        c1.metric(
            "Accuracy",
            f"{result.get('accuracy', 0):.1%}",
        )


        c2.metric(
            "Precision",
            f"{result.get('precision', 0):.1%}",
        )


        c3.metric(
            "Recall",
            f"{result.get('recall', 0):.1%}",
        )


        c4.metric(
            "F1 Score",
            f"{result.get('f1', 0):.1%}",
        )


        if (
            "feature_importance"
            in result
        ):

            st.subheader(
                "Feature Importance"
            )


            importance_chart(
                result[
                    "feature_importance"
                ]
            )


        return


    # -----------------------------------------------------
    # RANDOM FOREST REGRESSION
    # -----------------------------------------------------

    if (
        algorithm_name
        == "random-forest-regression"
        and isinstance(
            result,
            dict,
        )
        and function_name
        == "predict"
    ):

        c1, c2 = (
            st.columns(2)
        )


        c1.metric(
            "R² Score",
            f"{result.get('score', 0):.3f}",
        )


        c2.metric(
            "Average Error",
            f"{result.get('mae', 0):.3f}",
        )


        if (
            "feature_importance"
            in result
        ):

            st.subheader(
                "Feature Importance"
            )


            importance_chart(
                result[
                    "feature_importance"
                ]
            )


        return


    # -----------------------------------------------------
    # LINEAR REGRESSION
    # -----------------------------------------------------

    if (
        algorithm_name
        == "linear-regression"
        and isinstance(
            result,
            dict,
        )
    ):

        if (
            "score"
            in result
        ):

            st.metric(
                "R² Score",
                f"{result['score']:.3f}",
            )


        if (
            "y_predict"
            in result
        ):

            with st.expander(
                "Show predictions"
            ):

                st.dataframe(
                    pd.DataFrame
                    .from_dict(
                        result[
                            "y_predict"
                        ],
                        orient="index",
                    ),
                    width="stretch",
                )


        return


    # -----------------------------------------------------
    # LOGISTIC REGRESSION
    # -----------------------------------------------------

    if (
        algorithm_name
        == "logistic-regression"
        and isinstance(
            result,
            dict,
        )
        and function_name
        == "predict"
    ):

        c1, c2, c3, c4 = (
            st.columns(4)
        )


        c1.metric(
            "Accuracy",
            f"{result.get('accuracy', result.get('score', 0)):.1%}",
        )


        c2.metric(
            "Precision",
            f"{result.get('precision', 0):.1%}",
        )


        c3.metric(
            "Recall",
            f"{result.get('recall', 0):.1%}",
        )


        c4.metric(
            "F1 Score",
            f"{result.get('f1', 0):.1%}",
        )


        return


    # -----------------------------------------------------
    # ISOLATION FOREST
    # -----------------------------------------------------

    if (
        algorithm_name
        == "isolation-forest"
        and isinstance(
            result,
            dict,
        )
    ):

        if (
            "anomaly_count"
            in result
        ):

            c1, c2 = (
                st.columns(2)
            )


            c1.metric(
                "Rows Analysed",
                result.get(
                    "total_rows",
                    0,
                ),
            )


            c2.metric(
                "Anomalies",
                result.get(
                    "anomaly_count",
                    0,
                ),
            )


            if (
                "results"
                in result
            ):

                st.dataframe(
                    pd.DataFrame
                    .from_dict(
                        result[
                            "results"
                        ],
                        orient="index",
                    )
                    .head(100),
                    width="stretch",
                )


            return


    # -----------------------------------------------------
    # KMEANS
    # -----------------------------------------------------

    if (
        algorithm_name
        == "cluster"
        and isinstance(
            result,
            dict,
        )
    ):

        if (
            "labels"
            in result
        ):

            labels = (
                pd.DataFrame
                .from_dict(
                    result[
                        "labels"
                    ],
                    orient="index",
                )
            )


            st.subheader(
                "Cluster Assignments"
            )


            st.dataframe(
                labels.head(100),
                width="stretch",
            )


        if (
            "centers"
            in result
        ):

            centers = (
                pd.DataFrame
                .from_dict(
                    result[
                        "centers"
                    ],
                    orient="index",
                )
            )


            centers.index = [
                f"Group {position + 1}"

                for position, _
                in enumerate(
                    centers.index
                )
            ]


            st.subheader(
                "Cluster Profiles"
            )


            st.caption(
                "Center values are normalized "
                "between 0 and 1."
            )


            st.dataframe(
                centers.round(3),
                width="stretch",
            )


        return


    # -----------------------------------------------------
    # GENERIC
    # -----------------------------------------------------

    if isinstance(
        result,
        list,
    ):

        st.dataframe(
            pd.DataFrame(
                result
            ),
            width="stretch",
            hide_index=True,
        )


    elif isinstance(
        result,
        dict,
    ):

        try:

            st.dataframe(
                pd.DataFrame
                .from_dict(
                    result,
                    orient="index",
                ),
                width="stretch",
            )

        except Exception:

            st.json(
                result
            )


    else:

        st.write(
            result
        )


# =========================================================
# SIDEBAR
# =========================================================

if LOGO_PATH.exists():

    st.sidebar.image(
        str(
            LOGO_PATH
        ),
        width=155,
    )


st.sidebar.markdown(
    "### DPP Analytics"
)


page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Engagement",
        "Traceability",
        "Analytics Lab",
        "Dataset Explorer",
    ],
    label_visibility=
        "collapsed",
)


st.sidebar.divider()


st.sidebar.caption(
    "Mannstaedt / GMH Gruppe"
)


# =========================================================
# LOAD COMMON DATA
# =========================================================

try:

    sessions = (
        load_mock_sessions()
    )


    events = (
        load_mock_events()
    )


    traceability = (
        load_mock_traceability()
    )


except Exception as error:

    st.error(
        "Could not load the prototype datasets."
    )

    st.exception(
        error
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

show_header()


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="section-kicker">Overview</div>',
        unsafe_allow_html=True,
    )


    st.header(
        "DPP Performance Overview"
    )


    prototype_badge()


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(
        "Visitor Sessions",
        f"{len(sessions):,}",
    )


    c2.metric(
        "QR Traffic Share",
        f"{sessions['qr_entry'].mean():.1%}",
    )


    c3.metric(
        "Average Visit",
        f"{sessions['visit_duration'].mean():.0f} sec",
    )


    c4.metric(
        "Document Open Rate",
        f"{sessions['opened_document'].mean():.1%}",
    )


    st.divider()


    # =====================================================
    # QR VS WEB
    # =====================================================

    st.subheader(
        "QR vs Web Engagement"
    )


    source_summary = (
        sessions

        .groupby(
            "source"
        )

        .agg(

            Sessions=(
                "session_id",
                "count",
            ),

            Average_Visit=(
                "visit_duration",
                "mean",
            ),

            Average_Sections=(
                "sections_viewed",
                "mean",
            ),

            Document_Open_Rate=(
                "opened_document",
                "mean",
            ),
        )

        .reset_index()
    )


    source_summary[
        "source"
    ] = (
        source_summary[
            "source"
        ]
        .replace(
            {
                "qr":
                    "QR Entry",

                "web":
                    "Web Entry",
            }
        )
    )


    source_summary[
        "Average_Visit"
    ] = (
        source_summary[
            "Average_Visit"
        ]
        .round(1)
    )


    source_summary[
        "Average_Sections"
    ] = (
        source_summary[
            "Average_Sections"
        ]
        .round(2)
    )


    source_summary[
        "Document_Open_Rate"
    ] = (
        source_summary[
            "Document_Open_Rate"
        ]
        * 100
    ).round(1)


    source_summary = (
        source_summary.rename(
            columns={
                "source":
                    "Traffic Source",

                "Average_Visit":
                    "Average Visit (sec)",

                "Average_Sections":
                    "Sections / Visit",

                "Document_Open_Rate":
                    "Document Open Rate (%)",
            }
        )
    )


    st.dataframe(
        source_summary,
        width="stretch",
        hide_index=True,
    )


    # =====================================================
    # CHARTS
    # =====================================================

    left, right = (
        st.columns(2)
    )


    with left:

        st.subheader(
            "Most Viewed Sections"
        )


        section_df = (
            events[
                events[
                    "event_name"
                ]
                == "dpp-section-view"
            ][
                "section_name"
            ]
            .value_counts()
            .rename_axis(
                "Section"
            )
            .reset_index(
                name="Views"
            )
        )


        dark_bar(
            section_df,
            "Section:N",
            "Views:Q",
            "Views",
        )


    with right:

        st.subheader(
            "Most Opened Documents"
        )


        document_df = (
            events[
                events[
                    "event_name"
                ]
                == "dpp-document-open"
            ][
                "document_name"
            ]
            .dropna()
            .value_counts()
            .rename_axis(
                "Document"
            )
            .reset_index(
                name="Opens"
            )
        )


        dark_bar(
            document_df,
            "Document:N",
            "Opens:Q",
            "Opens",
        )


    st.divider()


    # =====================================================
    # TRACEABILITY
    # =====================================================

    st.subheader(
        "Traceability Snapshot"
    )


    high_risk = (
        traceability[
            traceability[
                "risk_score"
            ]
            >= 0.70
        ]
    )


    credential_issues = (
        traceability[
            traceability[
                "credential_status"
            ]
            != "valid"
        ]
    )


    t1, t2, t3 = (
        st.columns(3)
    )


    t1.metric(
        "Missing Links",
        int(
            traceability[
                "missing_link"
            ].sum()
        ),
    )


    t2.metric(
        "Credential Issues",
        len(
            credential_issues
        ),
    )


    t3.metric(
        "High-Risk Records",
        len(
            high_risk
        ),
    )


# =========================================================
# ENGAGEMENT
# =========================================================

elif page == "Engagement":

    st.markdown(
        '<div class="section-kicker">Engagement</div>',
        unsafe_allow_html=True,
    )


    st.header(
        "DPP Content & Visitor Engagement"
    )


    prototype_badge()


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(
        "Sessions",
        f"{len(sessions):,}",
    )


    c2.metric(
        "Average Visit",
        f"{sessions['visit_duration'].mean():.0f} sec",
    )


    c3.metric(
        "Sections / Visit",
        f"{sessions['sections_viewed'].mean():.1f}",
    )


    c4.metric(
        "Document Open Rate",
        f"{sessions['opened_document'].mean():.1%}",
    )


    st.divider()


    # =====================================================
    # SECTION PERFORMANCE
    # =====================================================

    st.subheader(
        "Section Performance"
    )


    section_performance = (
        prepare_section_performance_data()
    )


    if not section_performance.empty:

        display_sections = (
            section_performance.copy()
        )


        display_sections[
            "view_share"
        ] = (
            display_sections[
                "view_share"
            ]
            * 100
        ).round(1)


        display_sections = (
            display_sections.rename(
                columns={
                    "section_name":
                        "Section",

                    "views":
                        "Views",

                    "unique_sessions":
                        "Unique Sessions",

                    "view_share":
                        "Share of Views (%)",

                    "underused":
                        "Below Median Usage",
                }
            )
        )


        st.dataframe(
            display_sections,
            width="stretch",
            hide_index=True,
        )


        chart_df = (
            section_performance.rename(
                columns={
                    "section_name":
                        "Section",

                    "views":
                        "Views",
                }
            )
        )


        dark_bar(
            chart_df,
            "Section:N",
            "Views:Q",
            "Views",
            height=330,
        )


    # =====================================================
    # UNDERUSED
    # =====================================================

    st.subheader(
        "Content Requiring Attention"
    )


    if not section_performance.empty:

        underused = (
            section_performance[
                section_performance[
                    "underused"
                ]
            ]
            .sort_values(
                "views"
            )
        )


        if underused.empty:

            st.success(
                "No section is below the current "
                "median usage threshold."
            )


        else:

            display_underused = (
                underused.copy()
            )


            display_underused[
                "view_share"
            ] = (
                display_underused[
                    "view_share"
                ]
                * 100
            ).round(1)


            display_underused = (
                display_underused.rename(
                    columns={
                        "section_name":
                            "Section",

                        "views":
                            "Views",

                        "unique_sessions":
                            "Unique Sessions",

                        "view_share":
                            "Share of Views (%)",

                        "underused":
                            "Below Median Usage",
                    }
                )
            )


            st.dataframe(
                display_underused,
                width="stretch",
                hide_index=True,
            )


            weakest = (
                underused.iloc[
                    0
                ][
                    "section_name"
                ]
            )


            insight(
                "Content opportunity",
                (
                    f"<b>{weakest}</b> currently receives "
                    "the lowest usage among below-median sections."
                ),
            )


    # =====================================================
    # DOCUMENTS
    # =====================================================

    st.subheader(
        "Document Interaction"
    )


    document_df = (
        events[
            events[
                "event_name"
            ]
            == "dpp-document-open"
        ][
            "document_name"
        ]
        .dropna()
        .value_counts()
        .rename_axis(
            "Document"
        )
        .reset_index(
            name="Opens"
        )
    )


    dark_bar(
        document_df,
        "Document:N",
        "Opens:Q",
        "Opens",
    )


# =========================================================
# TRACEABILITY
# =========================================================

elif page == "Traceability":

    st.markdown(
        '<div class="section-kicker">Traceability</div>',
        unsafe_allow_html=True,
    )


    st.header(
        "Supply Chain Risk"
    )


    prototype_badge()


    high_risk = (
        traceability[
            traceability[
                "risk_score"
            ]
            >= 0.70
        ]
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(
        "Traceability Records",
        len(
            traceability
        ),
    )


    c2.metric(
        "Missing Links",
        int(
            traceability[
                "missing_link"
            ].sum()
        ),
    )


    c3.metric(
        "Credential Issues",
        len(
            traceability[
                traceability[
                    "credential_status"
                ]
                != "valid"
            ]
        ),
    )


    c4.metric(
        "High-Risk Records",
        len(
            high_risk
        ),
    )


    st.divider()


    supplier_df = (
        prepare_supplier_risk_data()
    )


    st.subheader(
        "Supplier Risk Overview"
    )


    supplier_chart = (
        supplier_df.rename(
            columns={
                "supplier_id":
                    "Supplier",

                "average_risk":
                    "Average Risk",
            }
        )
    )


    dark_bar(
        supplier_chart,
        "Supplier:N",
        "Average Risk:Q",
        "Average Risk",
    )


    st.subheader(
        "Records Requiring Attention"
    )


    st.dataframe(
        high_risk
        .sort_values(
            "risk_score",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )


# =========================================================
# ANALYTICS LAB
# =========================================================

elif page == "Analytics Lab":

    st.markdown(
        '<div class="section-kicker">Analytics Lab</div>',
        unsafe_allow_html=True,
    )


    st.header(
        "Machine Learning & Advanced Analytics"
    )


    prototype_badge()


    analytics_mode = (
        st.radio(
            "Analytics Mode",
            [
                "Guided Business Analysis",
                "Advanced Model Explorer",
            ],
            horizontal=True,
        )
    )


    # =====================================================
    # GUIDED MODE
    # =====================================================

    if (
        analytics_mode
        == "Guided Business Analysis"
    ):

        st.markdown(
            """
<div class="mode-card">

<div class="mode-title">
Guided Business Analysis
</div>

<div class="mode-description">
Choose a business question. The dashboard selects
the machine-learning method automatically.
</div>

</div>
""",
            unsafe_allow_html=True,
        )


        BUSINESS_ANALYSES = [

            (
                "What drives document opening? "
                "[Random Forest Classifier]"
            ),

            (
                "Which engagement sessions look unusual? "
                "[Isolation Forest]"
            ),

            (
                "Which records are likely to be high "
                "traceability risk? "
                "[Random Forest Classifier]"
            ),

            (
                "What drives traceability risk? "
                "[Random Forest Regressor]"
            ),

            (
                "Which suppliers have similar risk profiles? "
                "[K-Means Clustering]"
            ),
        ]


        choice = (
            st.selectbox(
                "What would you like to understand?",
                BUSINESS_ANALYSES,
            )
        )


        # -------------------------------------------------
        # DOCUMENT OPEN
        # -------------------------------------------------

        if choice.startswith(
            "What drives document opening?"
        ):

            st.info(
                "Identify which engagement variables "
                "are most useful for predicting "
                "document opening."
            )


            if st.button(
                "Analyse Document-Open Drivers",
                type="primary",
                use_container_width=True,
            ):

                ml_df = (
                    prepare_engagement_ml_data(
                        sessions
                    )
                )


                features = [
                    "qr_entry",
                    "visit_duration",
                    "sections_viewed",
                ]


                output = (
                    run_engine(

                        dataframe=
                            ml_df,

                        algorithm_name=
                            "random-forest-classification",

                        function_name=
                            "predict",

                        included_columns=
                            features,

                        target_columns=[
                            "opened_document"
                        ],

                        config={
                            "random_state":
                                42,

                            "test_size":
                                0.20,

                            "n_estimators":
                                300,
                        },
                    )
                )


                result = (
                    output[
                        "result"
                    ]
                )


                c1, c2, c3, c4 = (
                    st.columns(4)
                )


                c1.metric(
                    "Accuracy",
                    f"{result['accuracy']:.1%}",
                )


                c2.metric(
                    "Precision",
                    f"{result['precision']:.1%}",
                )


                c3.metric(
                    "Recall",
                    f"{result['recall']:.1%}",
                )


                c4.metric(
                    "F1 Score",
                    f"{result['f1']:.1%}",
                )


                st.subheader(
                    "Strongest Predictive Signals"
                )


                importance_chart(
                    result[
                        "feature_importance"
                    ]
                )


        # -------------------------------------------------
        # ANOMALIES
        # -------------------------------------------------

        elif choice.startswith(
            "Which engagement sessions look unusual?"
        ):

            st.info(
                "Detect engagement sessions that differ "
                "substantially from typical behaviour."
            )


            contamination = (
                st.slider(
                    "Expected unusual-session share",
                    0.01,
                    0.20,
                    0.05,
                    0.01,
                )
            )


            if st.button(
                "Detect Unusual Sessions",
                type="primary",
                use_container_width=True,
            ):

                ml_df = (
                    prepare_engagement_ml_data(
                        sessions
                    )
                )


                features = [
                    "qr_entry",
                    "visit_duration",
                    "sections_viewed",
                    "documents_opened",
                ]


                output = (
                    run_engine(

                        dataframe=
                            ml_df,

                        algorithm_name=
                            "isolation-forest",

                        function_name=
                            "predict",

                        included_columns=
                            features,

                        config={
                            "random_state":
                                42,

                            "n_estimators":
                                300,

                            "contamination":
                                contamination,
                        },
                    )
                )


                result = (
                    output[
                        "result"
                    ]
                )


                c1, c2 = (
                    st.columns(2)
                )


                c1.metric(
                    "Sessions Analysed",
                    result[
                        "total_rows"
                    ],
                )


                c2.metric(
                    "Unusual Sessions",
                    result[
                        "anomaly_count"
                    ],
                )


                anomaly_df = (
                    pd.DataFrame
                    .from_dict(
                        result[
                            "results"
                        ],
                        orient="index",
                    )
                )


                anomaly_df.index = (
                    pd.to_numeric(
                        anomaly_df.index,
                        errors="coerce",
                    )
                )


                anomaly_df = (
                    anomaly_df[
                        anomaly_df.index
                        .notna()
                    ]
                )


                anomaly_df.index = (
                    anomaly_df.index
                    .astype(int)
                )


                anomalies = (
                    anomaly_df[
                        anomaly_df[
                            "is_anomaly"
                        ]
                        == 1
                    ]
                )


                valid_indexes = (
                    anomalies.index
                    .intersection(
                        sessions.index
                    )
                )


                original = (
                    sessions.loc[
                        valid_indexes
                    ]
                    .copy()
                )


                original[
                    "anomaly_score"
                ] = (
                    anomalies.loc[
                        valid_indexes,
                        "anomaly_score"
                    ]
                    .values
                )


                st.dataframe(
                    original
                    .sort_values(
                        "anomaly_score"
                    )
                    .head(25),
                    width="stretch",
                    hide_index=True,
                )


        # -------------------------------------------------
        # HIGH RISK CLASSIFIER
        # -------------------------------------------------

        elif choice.startswith(
            "Which records are likely"
        ):

            st.info(
                "Estimate which traceability records "
                "may require closer review."
            )


            if st.button(
                "Analyse High-Risk Prediction",
                type="primary",
                use_container_width=True,
            ):

                ml_df = (
                    prepare_traceability_ml_data(
                        traceability
                    )
                )


                features = [
                    column

                    for column
                    in ml_df.columns

                    if (
                        column
                        not in [
                            "risk_score",
                            "high_risk",
                        ]
                        and
                        pd.api.types
                        .is_numeric_dtype(
                            ml_df[
                                column
                            ]
                        )
                    )
                ]


                output = (
                    run_engine(

                        dataframe=
                            ml_df,

                        algorithm_name=
                            "random-forest-classification",

                        function_name=
                            "predict",

                        included_columns=
                            features,

                        target_columns=[
                            "high_risk"
                        ],

                        config={
                            "random_state":
                                42,

                            "test_size":
                                0.20,

                            "n_estimators":
                                300,
                        },
                    )
                )


                result = (
                    output[
                        "result"
                    ]
                )


                c1, c2, c3, c4 = (
                    st.columns(4)
                )


                c1.metric(
                    "Accuracy",
                    f"{result['accuracy']:.1%}",
                )


                c2.metric(
                    "Precision",
                    f"{result['precision']:.1%}",
                )


                c3.metric(
                    "Recall",
                    f"{result['recall']:.1%}",
                )


                c4.metric(
                    "F1 Score",
                    f"{result['f1']:.1%}",
                )


                importance_chart(
                    result[
                        "feature_importance"
                    ]
                )


        # -------------------------------------------------
        # RISK DRIVERS
        # -------------------------------------------------

        elif choice.startswith(
            "What drives traceability risk?"
        ):

            st.info(
                "Identify which traceability attributes "
                "carry predictive information for the "
                "prototype risk score."
            )


            if st.button(
                "Analyse Risk Drivers",
                type="primary",
                use_container_width=True,
            ):

                ml_df = (
                    prepare_traceability_ml_data(
                        traceability
                    )
                )


                features = [
                    column

                    for column
                    in ml_df.columns

                    if (
                        column
                        not in [
                            "risk_score",
                            "high_risk",
                        ]
                        and
                        pd.api.types
                        .is_numeric_dtype(
                            ml_df[
                                column
                            ]
                        )
                    )
                ]


                output = (
                    run_engine(

                        dataframe=
                            ml_df,

                        algorithm_name=
                            "random-forest-regression",

                        function_name=
                            "predict",

                        included_columns=
                            features,

                        target_columns=[
                            "risk_score"
                        ],

                        config={
                            "random_state":
                                42,

                            "test_size":
                                0.20,

                            "n_estimators":
                                300,
                        },
                    )
                )


                result = (
                    output[
                        "result"
                    ]
                )


                c1, c2 = (
                    st.columns(2)
                )


                c1.metric(
                    "R² Score",
                    f"{result['score']:.3f}",
                )


                c2.metric(
                    "Average Risk Error",
                    f"{result['mae']:.3f}",
                )


                importance_chart(
                    result[
                        "feature_importance"
                    ]
                )


        # -------------------------------------------------
        # SUPPLIER CLUSTERING
        # -------------------------------------------------

        elif choice.startswith(
            "Which suppliers"
        ):

            supplier_df = (
                prepare_supplier_risk_data()
            )


            max_clusters = min(
                6,
                len(
                    supplier_df
                ),
            )


            cluster_count = (
                st.slider(
                    "Number of supplier groups",
                    2,
                    max_clusters,
                    min(
                        3,
                        max_clusters,
                    ),
                )
            )


            if st.button(
                "Group Suppliers",
                type="primary",
                use_container_width=True,
            ):

                features = [
                    "records",
                    "average_risk",
                    "missing_link_rate",
                    "credential_issue_rate",
                ]


                output = (
                    run_engine(

                        dataframe=
                            supplier_df,

                        algorithm_name=
                            "cluster",

                        function_name=
                            "predict",

                        included_columns=
                            features,

                        config={
                            "random_state":
                                42,

                            "n_cluster":
                                cluster_count,
                        },
                    )
                )


                result = (
                    output[
                        "result"
                    ]
                )


                labels = (
                    pd.DataFrame
                    .from_dict(
                        result[
                            "labels"
                        ],
                        orient="index",
                    )
                    .reset_index(
                        drop=True
                    )
                )


                grouped = (
                    supplier_df
                    .copy()
                    .reset_index(
                        drop=True
                    )
                )


                grouped[
                    "risk_group"
                ] = (
                    labels[
                        "label"
                    ]
                    .astype(int)
                    .values
                    + 1
                )


                st.dataframe(
                    grouped,
                    width="stretch",
                    hide_index=True,
                )


                centers = (
                    pd.DataFrame
                    .from_dict(
                        result[
                            "centers"
                        ],
                        orient="index",
                    )
                )


                centers.index = [
                    f"Group {position + 1}"

                    for position, _
                    in enumerate(
                        centers.index
                    )
                ]


                st.caption(
                    "Cluster center values are normalized "
                    "between 0 and 1."
                )


                st.dataframe(
                    centers.round(3),
                    width="stretch",
                )


    # =====================================================
    # ADVANCED MODEL EXPLORER
    # =====================================================

    else:

        st.markdown(
            """
<div class="mode-card">

<div class="mode-title">
Advanced Model Explorer
</div>

<div class="mode-description">
Choose the dataset, algorithm, function, target and
features manually. This exposes the generic
AnalyticEngine workflow.
</div>

</div>
""",
            unsafe_allow_html=True,
        )


        # =================================================
        # 1. DATASET
        # =================================================

        datasets = (
            get_available_datasets()
        )


        dataset_label = (
            st.selectbox(
                "1. Choose Dataset",
                list(
                    datasets.keys()
                ),
                key=
                    "advanced_dataset",
            )
        )


        dataset_key = (
            datasets[
                dataset_label
            ]
        )


        raw_df = (
            load_dataset(
                dataset_key
            )
        )


        # =================================================
        # AUTOMATIC DATA PREPARATION
        # =================================================

        preparation = (
            prepare_dataset_for_ml(
                dataset_key=
                    dataset_key,

                raw_df=
                    raw_df,
            )
        )


        ml_df = (
            preparation[
                "dataframe"
            ]
        )


        preparation_summary = (
            preparation[
                "summary"
            ]
        )


        # THIS IS THE NEW SECTION

        show_data_preparation_summary(
            preparation_summary
        )


        # =================================================
        # NUMERIC COLUMNS
        # =================================================

        numeric_columns = (
            ml_df
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )


        if not numeric_columns:

            st.warning(
                "No model-ready numeric columns "
                "remain after data preparation."
            )

            st.stop()


        # =================================================
        # 2. MODEL
        # =================================================

        model_label = (
            st.selectbox(
                "2. Choose Model",
                list(
                    MODEL_OPTIONS.keys()
                ),
                key=
                    "advanced_model",
            )
        )


        algorithm_name = (
            MODEL_OPTIONS[
                model_label
            ]
        )


        st.markdown(
            f"""
<div class="model-badge">
{MODEL_FRIENDLY_NAMES.get(algorithm_name, algorithm_name)}
</div>
""",
            unsafe_allow_html=True,
        )


        # =================================================
        # 3. FUNCTION
        # =================================================

        available_functions = (
            FUNCTION_OPTIONS.get(
                algorithm_name,
                []
            )
        )


        # Generic explorer does not need scenario
        # prediction yet.

        available_functions = [
            function

            for function
            in available_functions

            if function
            != "predict_input"
        ]


        function_name = (
            st.selectbox(
                "3. Choose Function",
                available_functions,
                format_func=lambda value:
                    FUNCTION_FRIENDLY_NAMES.get(
                        value,
                        value,
                    ),
                key=
                    "advanced_function",
            )
        )


        # =================================================
        # MODEL TYPE
        # =================================================

        supervised_models = {
            "linear-regression",
            "logistic-regression",
            "random-forest-regression",
            "random-forest-classification",
        }


        classification_models = {
            "logistic-regression",
            "random-forest-classification",
        }


        target_column = None


        # =================================================
        # 4. TARGET
        # =================================================

        if (
            algorithm_name
            in supervised_models
        ):

            target_column = (
                st.selectbox(
                    "4. Choose Target",
                    numeric_columns,
                    format_func=
                        friendly_feature,
                    key=
                        "advanced_target",
                )
            )


            if (
                algorithm_name
                in classification_models
            ):

                target_unique = (
                    ml_df[
                        target_column
                    ]
                    .nunique(
                        dropna=True
                    )
                )


                if target_unique > 20:

                    st.warning(
                        "This target contains many unique "
                        "values. A classification model usually "
                        "works better with categorical or "
                        "binary targets."
                    )


        # =================================================
        # FEATURES
        # =================================================

        available_features = [
            column

            for column
            in numeric_columns

            if column
            != target_column
        ]


        default_features = (
            available_features[
                :min(
                    5,
                    len(
                        available_features
                    ),
                )
            ]
        )


        selected_features = (
            st.multiselect(
                (
                    "5. Choose Input Features"
                    if target_column
                    else
                    "4. Choose Input Features"
                ),
                available_features,
                default=
                    default_features,
                format_func=
                    friendly_feature,
                key=
                    "advanced_features",
            )
        )


        # =================================================
        # CONFIG
        # =================================================

        config = {
            "random_state":
                42,
        }


        if (
            algorithm_name
            in supervised_models
        ):

            config[
                "test_size"
            ] = (
                st.slider(
                    "Test Data Share",
                    0.10,
                    0.40,
                    0.20,
                    0.05,
                    key=
                        "advanced_test_size",
                )
            )


        if (
            algorithm_name
            in [
                "random-forest-regression",
                "random-forest-classification",
                "isolation-forest",
            ]
        ):

            config[
                "n_estimators"
            ] = (
                st.slider(
                    "Number of Trees",
                    100,
                    500,
                    300,
                    50,
                    key=
                        "advanced_estimators",
                )
            )


        if (
            algorithm_name
            == "cluster"
        ):

            config[
                "n_cluster"
            ] = (
                st.slider(
                    "Number of Clusters",
                    2,
                    10,
                    3,
                    key=
                        "advanced_clusters",
                )
            )


        if (
            algorithm_name
            == "isolation-forest"
        ):

            config[
                "contamination"
            ] = (
                st.slider(
                    "Expected Anomaly Share",
                    0.01,
                    0.20,
                    0.05,
                    0.01,
                    key=
                        "advanced_contamination",
                )
            )


        # =================================================
        # REQUEST SUMMARY
        # =================================================

        st.divider()


        st.subheader(
            "Engine Request Summary"
        )


        col1, col2 = (
            st.columns(2)
        )


        with col1:

            st.write(
                f"**Dataset:** "
                f"{dataset_label}"
            )


            st.write(
                f"**Model:** "
                f"{model_label}"
            )


            st.write(
                "**Function:** "
                + FUNCTION_FRIENDLY_NAMES.get(
                    function_name,
                    function_name,
                )
            )


        with col2:

            if target_column:

                st.write(
                    "**Target:** "
                    + friendly_feature(
                        target_column
                    )
                )


            st.write(
                "**Features:** "
                + (
                    ", ".join(
                        friendly_feature(
                            column
                        )

                        for column
                        in selected_features
                    )

                    if selected_features

                    else "None"
                )
            )


        # =================================================
        # RUN
        # =================================================

        if st.button(
            "Run Advanced Analysis",
            type="primary",
            use_container_width=True,
            key=
                "run_advanced",
        ):

            if not selected_features:

                st.error(
                    "Choose at least one input feature."
                )

                st.stop()


            try:

                with st.spinner(
                    "Running AnalyticEngine..."
                ):

                    output = (
                        run_engine(

                            # IMPORTANT:
                            # send prepared data, not raw data

                            dataframe=
                                ml_df,

                            algorithm_name=
                                algorithm_name,

                            function_name=
                                function_name,

                            included_columns=
                                selected_features,

                            target_columns=(
                                [
                                    target_column
                                ]

                                if target_column

                                else []
                            ),

                            config=
                                config,

                            function_input=
                                {},
                        )
                    )


                st.success(
                    "Analysis completed."
                )


                result = (
                    output[
                        "result"
                    ]
                )


                st.subheader(
                    "Machine Learning Result"
                )


                display_advanced_result(
                    result,
                    algorithm_name,
                    function_name,
                )


                with st.expander(
                    "AnalyticEngine setup_data"
                ):

                    st.json(
                        output[
                            "setup_data"
                        ]
                    )


                with st.expander(
                    "Raw engine result"
                ):

                    st.write(
                        result
                    )


            except Exception as error:

                st.error(
                    "The selected analysis could not run."
                )


                with st.expander(
                    "Technical error"
                ):

                    st.exception(
                        error
                    )


# =========================================================
# DATASET EXPLORER
# =========================================================

elif page == "Dataset Explorer":

    st.markdown(
        '<div class="section-kicker">Data</div>',
        unsafe_allow_html=True,
    )


    st.header(
        "Dataset Explorer"
    )


    datasets = (
        get_available_datasets()
    )


    selected_dataset = (
        st.selectbox(
            "Dataset",
            list(
                datasets.keys()
            ),
        )
    )


    dataset_key = (
        datasets[
            selected_dataset
        ]
    )


    df = (
        load_dataset(
            dataset_key
        )
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    c1.metric(
        "Rows",
        len(
            df
        ),
    )


    c2.metric(
        "Columns",
        len(
            df.columns
        ),
    )


    c3.metric(
        "Missing Values",
        int(
            df
            .isna()
            .sum()
            .sum()
        ),
    )


    st.subheader(
        "Data Preview"
    )


    st.dataframe(
        df.head(250),
        width="stretch",
        hide_index=True,
    )


    # =====================================================
    # OPTIONAL PREPARATION PREVIEW
    # =====================================================

    with st.expander(
        "Preview ML data preparation"
    ):

        preparation = (
            prepare_dataset_for_ml(
                dataset_key=
                    dataset_key,

                raw_df=
                    df,
            )
        )


        show_data_preparation_summary(
            preparation[
                "summary"
            ]
        )


        st.subheader(
            "Prepared ML Data"
        )


        st.dataframe(
            preparation[
                "dataframe"
            ]
            .head(100),
            width="stretch",
            hide_index=True,
        )