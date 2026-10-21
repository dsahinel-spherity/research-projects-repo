from __future__ import annotations

from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.services.dpp_filter_service import CUSTOM_EVENTS, filter_gmh_events, filter_selected_dpp

from app.services.analytics_service import (
    document_performance,
    dpp_performance,
    real_company_kpis,
    section_performance,
)

from app.services.data_service import (
    get_available_datasets,
    load_dataset,
    load_mock_journeys,
    load_real_analytics,
)

from app.services.data_preparation_service import (
    prepare_dataset_for_ml,
)

from app.services.engine_service import (
    FUNCTION_OPTIONS,
    MODEL_OPTIONS,
    run_engine,
)

from app.services.journey_service import (
    build_user_journeys,
    entry_exit_sections,
    journey_kpis,
    journey_transitions,
    section_document_conversion,
    top_journeys,
)


# =========================================================
# ORGANIZATION
# =========================================================

GMH_ORGANIZATION = (
    "gmh-gruppe-d034406b"
)


# =========================================================
# PAGE
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
# PATH
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

h1, h2, h3 {
    color: #F4F4F5 !important;
}

p {
    color: #B1B3B8;
}

[data-testid="stMetric"] {
    background: #18191D;
    border: 1px solid #292B30;
    border-radius: 10px;
    padding: 16px 18px;
}

[data-testid="stMetricLabel"] {
    color: #8D9097;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .04em;
}

[data-testid="stMetricValue"] {
    color: white;
    font-size: 28px;
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border: 1px solid #292B30;
    border-radius: 9px;
}

[data-testid="stExpander"] {
    background: #16171A;
    border: 1px solid #292B30;
    border-radius: 9px;
}

.stButton > button {
    background: #E30613;
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 45px;
    font-weight: 650;
}

.stButton > button:hover {
    background: #B90510;
    color: white;
}

.mannstaedt-title {
    color: white;
    font-size: 31px;
    font-weight: 720;
}

.mannstaedt-subtitle {
    color: #8F929A;
    margin-top: 5px;
    margin-bottom: 30px;
}

.section-kicker {
    color: #E30613;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.real-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 11px;
    background: rgba(25, 180, 90, .08);
    border: 1px solid rgba(25, 180, 90, .35);
    border-radius: 100px;
    color: #D1D2D5;
    font-size: 12px;
    margin-bottom: 22px;
}

.real-dot {
    width: 7px;
    height: 7px;
    background: #19B45A;
    border-radius: 50%;
}

.prototype-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 11px;
    background: rgba(227, 6, 19, .08);
    border: 1px solid rgba(227, 6, 19, .32);
    border-radius: 100px;
    color: #D1D2D5;
    font-size: 12px;
    margin-bottom: 22px;
}

.prototype-dot {
    width: 7px;
    height: 7px;
    background: #E30613;
    border-radius: 50%;
}

.insight-card {
    background: #18191D;
    border: 1px solid #292B30;
    border-left: 3px solid #E30613;
    border-radius: 8px;
    padding: 18px;
    margin: 15px 0;
    color: #CACBCF;
}

.insight-card b {
    color: white;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# UI HELPERS
# =========================================================

def show_header():

    st.markdown(
        """
<div class="mannstaedt-title">
Mannstaedt DPP Analytics
</div>

<div class="mannstaedt-subtitle">
GMH Gruppe Digital Product Passport Intelligence
</div>
""",
        unsafe_allow_html=True,
    )


def real_badge():

    st.markdown(
        """
<div class="real-badge">
<span class="real-dot"></span>
Real GMH Umami data
</div>
""",
        unsafe_allow_html=True,
    )


def prototype_badge():

    st.markdown(
        """
<div class="prototype-badge">
<span class="prototype-dot"></span>
Synthetic prototype data — ML only
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


def seconds_label(
    seconds,
):

    seconds = float(
        seconds or 0
    )

    if seconds < 60:

        return (
            f"{seconds:.0f} sec"
        )

    return (
        f"{seconds / 60:.1f} min"
    )


# =========================================================
# QR VS WEB ENGAGEMENT
# =========================================================

def qr_vs_web_engagement_analysis(
    journeys_df,
):

    empty_result = {
        "table":
            pd.DataFrame(),

        "qr_journeys":
            0,

        "web_journeys":
            0,

        "qr_document_conversion":
            0.0,

        "web_document_conversion":
            0.0,

        "qr_avg_sections":
            0.0,

        "web_avg_sections":
            0.0,

        "qr_avg_duration":
            0.0,

        "web_avg_duration":
            0.0,

        "conclusion":
            (
                "There is not enough journey data yet "
                "to compare QR and web visitors."
            ),
    }

    if (
        journeys_df is None
        or journeys_df.empty
    ):

        return empty_result

    journeys = (
        journeys_df.copy()
    )

    if (
        "source"
        not in journeys.columns
    ):

        return empty_result

    journeys[
        "source"
    ] = (
        journeys[
            "source"
        ]
        .fillna("web")
        .astype(str)
        .str.lower()
    )

    journeys[
        "source_group"
    ] = (
        journeys[
            "source"
        ]
        .apply(
            lambda value:
                "QR"
                if value == "qr"
                else "Web"
        )
    )

    rows = []

    for source_name in [
        "QR",
        "Web",
    ]:

        group = (
            journeys[
                journeys[
                    "source_group"
                ]
                == source_name
            ]
            .copy()
        )

        journey_count = (
            len(
                group
            )
        )

        if journey_count:

            document_conversion = (
                pd.to_numeric(
                    group[
                        "opened_document"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .astype(float)
                .mean()
                * 100
            )

            avg_sections = (
                pd.to_numeric(
                    group[
                        "section_count"
                    ],
                    errors="coerce",
                )
                .mean()
            )

            avg_duration = (
                pd.to_numeric(
                    group[
                        "duration_seconds"
                    ],
                    errors="coerce",
                )
                .mean()
            )

        else:

            document_conversion = 0.0
            avg_sections = 0.0
            avg_duration = 0.0

        rows.append(
            {
                "Source":
                    source_name,

                "Journeys":
                    int(
                        journey_count
                    ),

                "Document Conversion (%)":
                    round(
                        float(
                            document_conversion
                            or 0
                        ),
                        1,
                    ),

                "Avg Sections":
                    round(
                        float(
                            avg_sections
                            if pd.notna(
                                avg_sections
                            )
                            else 0
                        ),
                        2,
                    ),

                "Avg Duration (sec)":
                    round(
                        float(
                            avg_duration
                            if pd.notna(
                                avg_duration
                            )
                            else 0
                        ),
                        1,
                    ),
            }
        )

    comparison = (
        pd.DataFrame(
            rows
        )
    )

    qr_row = (
        comparison[
            comparison[
                "Source"
            ]
            == "QR"
        ]
        .iloc[0]
    )

    web_row = (
        comparison[
            comparison[
                "Source"
            ]
            == "Web"
        ]
        .iloc[0]
    )

    qr_journeys = int(
        qr_row[
            "Journeys"
        ]
    )

    web_journeys = int(
        web_row[
            "Journeys"
        ]
    )

    qr_conversion = float(
        qr_row[
            "Document Conversion (%)"
        ]
    )

    web_conversion = float(
        web_row[
            "Document Conversion (%)"
        ]
    )

    qr_sections = float(
        qr_row[
            "Avg Sections"
        ]
    )

    web_sections = float(
        web_row[
            "Avg Sections"
        ]
    )

    qr_duration = float(
        qr_row[
            "Avg Duration (sec)"
        ]
    )

    web_duration = float(
        web_row[
            "Avg Duration (sec)"
        ]
    )

    # =====================================================
    # HUMAN INTERPRETATION
    # =====================================================

    if (
        qr_journeys == 0
        or web_journeys == 0
    ):

        conclusion = (
            "Both QR and web journeys are required before "
            "their engagement can be compared."
        )

    else:

        signals = []

        # -------------------------------------------------
        # DOCUMENT CONVERSION
        # -------------------------------------------------

        conversion_difference = (
            qr_conversion
            - web_conversion
        )

        if abs(
            conversion_difference
        ) >= 5:

            if (
                conversion_difference
                > 0
            ):

                signals.append(
                    (
                        f"QR visitors have a "
                        f"{abs(conversion_difference):.1f} "
                        f"percentage-point higher document-open rate"
                    )
                )

            else:

                signals.append(
                    (
                        f"Web visitors have a "
                        f"{abs(conversion_difference):.1f} "
                        f"percentage-point higher document-open rate"
                    )
                )

        # -------------------------------------------------
        # JOURNEY DEPTH
        # Correct denominator:
        # compare the larger value against the smaller/base
        # group named in the sentence.
        # -------------------------------------------------

        if (
            qr_sections > 0
            and web_sections > 0
        ):

            if (
                web_sections
                > qr_sections
            ):

                section_difference = (
                    (
                        web_sections
                        - qr_sections
                    )
                    / qr_sections
                    * 100
                )

                if (
                    section_difference
                    >= 10
                ):

                    signals.append(
                        (
                            f"Web visitors explore about "
                            f"{section_difference:.0f}% more sections "
                            f"than QR visitors"
                        )
                    )

            elif (
                qr_sections
                > web_sections
            ):

                section_difference = (
                    (
                        qr_sections
                        - web_sections
                    )
                    / web_sections
                    * 100
                )

                if (
                    section_difference
                    >= 10
                ):

                    signals.append(
                        (
                            f"QR visitors explore about "
                            f"{section_difference:.0f}% more sections "
                            f"than web visitors"
                        )
                    )

        # -------------------------------------------------
        # JOURNEY DURATION
        # Correct denominator:
        # compare larger duration against smaller/base.
        # -------------------------------------------------

        if (
            qr_duration > 0
            and web_duration > 0
        ):

            if (
                web_duration
                > qr_duration
            ):

                duration_difference = (
                    (
                        web_duration
                        - qr_duration
                    )
                    / qr_duration
                    * 100
                )

                if (
                    duration_difference
                    >= 10
                ):

                    signals.append(
                        (
                            f"Web journeys last about "
                            f"{duration_difference:.0f}% longer "
                            f"than QR journeys"
                        )
                    )

            elif (
                qr_duration
                > web_duration
            ):

                duration_difference = (
                    (
                        qr_duration
                        - web_duration
                    )
                    / web_duration
                    * 100
                )

                if (
                    duration_difference
                    >= 10
                ):

                    signals.append(
                        (
                            f"QR journeys last about "
                            f"{duration_difference:.0f}% longer "
                            f"than web journeys"
                        )
                    )

        # -------------------------------------------------
        # FINAL CONCLUSION
        # -------------------------------------------------

        if signals:

            conclusion = (
                "The current GMH data suggests different "
                "engagement patterns between QR and web visitors. "
                + ". ".join(
                    signals
                )
                + "."
            )

        else:

            conclusion = (
                "The current GMH data does not show a strong "
                "engagement difference between QR and web visitors "
                "across document opening, journey depth and duration."
            )

        if (
            min(
                qr_journeys,
                web_journeys,
            )
            < 10
        ):

            conclusion += (
                " The sample is still small, so this should "
                "be treated as an early indication rather "
                "than a stable conclusion."
            )

    return {
        "table":
            comparison,

        "qr_journeys":
            qr_journeys,

        "web_journeys":
            web_journeys,

        "qr_document_conversion":
            qr_conversion,

        "web_document_conversion":
            web_conversion,

        "qr_avg_sections":
            qr_sections,

        "web_avg_sections":
            web_sections,

        "qr_avg_duration":
            qr_duration,

        "web_avg_duration":
            web_duration,

        "conclusion":
            conclusion,
    }


# =========================================================
# CHART
# =========================================================

def dark_bar(
    df,
    x,
    y,
    y_title,
    height=320,
):

    if (
        df is None
        or df.empty
    ):

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
                sort="-y",
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
                height
        )
        .configure_view(
            fill=
                CARD_BACKGROUND,

            strokeOpacity=
                0,
        )
        .configure(
            background=
                CARD_BACKGROUND
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
        theme=None,
    )


# =========================================================
# PREPARATION SUMMARY
# =========================================================

def show_preparation_summary(
    preparation,
    selected_features=None,
):

    summary = (
        preparation[
            "summary"
        ]
    )

    st.subheader(
        "Data Cleaning & Preparation"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Original Rows",
        summary.get(
            "original_rows",
            0,
        ),
    )

    c2.metric(
        "Final Rows",
        summary.get(
            "final_rows",
            0,
        ),
    )

    c3.metric(
        "Duplicates Removed",
        summary.get(
            "duplicates_removed",
            0,
        ),
    )

    c4.metric(
        "Missing Values Filled",
        summary.get(
            "missing_values_filled",
            0,
        ),
    )

    c5, c6 = (
        st.columns(2)
    )

    c5.metric(
        "Infinite Values Cleaned",
        summary.get(
            "infinite_values_cleaned",
            0,
        ),
    )

    c6.metric(
        "Model-Ready Columns",
        summary.get(
            "final_columns",
            0,
        ),
    )

    with st.expander(
        "What happened to the data?"
    ):

        st.write(
            "**Columns containing missing values:**"
        )

        st.write(
            summary.get(
                "columns_with_missing",
                [],
            )
            or "None"
        )

        st.write(
            "**Categorical columns encoded:**"
        )

        st.write(
            summary.get(
                "encoded_columns",
                [],
            )
            or "None"
        )

        st.write(
            "**Columns excluded from modelling:**"
        )

        st.write(
            summary.get(
                "excluded_columns",
                [],
            )
            or "None"
        )

        if selected_features:

            st.write(
                "**Features used by this model:**"
            )

            st.write(
                selected_features
            )


# =========================================================
# MODEL RESULT DISPLAY
# =========================================================

def display_result(
    result,
    algorithm_name,
    context_df=None,
):

    if not isinstance(
        result,
        dict,
    ):

        st.write(
            result
        )

        return

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if (
        algorithm_name
        in {
            "random-forest-classification",
            "logistic-regression",
        }
    ):

        classification_type = (
            result.get(
                "classification_type"
            )
        )

        class_count = (
            result.get(
                "class_count"
            )
        )

        metric_average = (
            result.get(
                "metric_average"
            )
        )

        if classification_type:

            if (
                classification_type
                == "multiclass"
            ):

                st.info(
                    (
                        f"Multiclass classification — "
                        f"{class_count} classes. "
                        f"Precision, recall and F1 are "
                        f"{metric_average}-averaged."
                    )
                )

            else:

                st.info(
                    "Binary classification."
                )

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

        importance = (
            result.get(
                "feature_importance"
            )
        )

        if importance:

            importance_df = (
                pd.DataFrame(
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
            )

            dark_bar(
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False,
                )
                .head(10),

                "Feature:N",
                "Importance:Q",
                "Relative Importance",
            )

        return

    # =====================================================
    # REGRESSION
    # =====================================================

    if (
        algorithm_name
        in {
            "random-forest-regression",
            "linear-regression",
        }
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

        importance = (
            result.get(
                "feature_importance"
            )
        )

        if importance:

            importance_df = (
                pd.DataFrame(
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
            )

            dark_bar(
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False,
                )
                .head(10),

                "Feature:N",
                "Importance:Q",
                "Relative Importance",
            )

        return

    # =====================================================
    # ISOLATION FOREST
    # =====================================================

    if (
        algorithm_name
        == "isolation-forest"
    ):

        total = (
            result.get(
                "total_rows",
                0,
            )
        )

        anomaly_count = (
            result.get(
                "anomaly_count",
                0,
            )
        )

        anomaly_rate = (
            anomaly_count
            / total
            * 100
            if total
            else 0
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        c1.metric(
            "Rows Analysed",
            total,
        )

        c2.metric(
            "Unusual Journeys",
            anomaly_count,
        )

        c3.metric(
            "Unusual Share",
            f"{anomaly_rate:.1f}%",
        )

        if (
            "results"
            not in result
        ):

            return

        anomaly_result = (
            pd.DataFrame
            .from_dict(
                result[
                    "results"
                ],
                orient="index",
            )
        )

        anomaly_result.index = (
            pd.to_numeric(
                anomaly_result.index,
                errors="coerce",
            )
        )

        anomaly_result = (
            anomaly_result[
                anomaly_result.index
                .notna()
            ]
            .copy()
        )

        anomaly_result.index = (
            anomaly_result.index
            .astype(int)
        )

        if (
            context_df is not None
            and not context_df.empty
        ):

            context = (
                context_df
                .copy()
                .reset_index(
                    drop=True
                )
            )

            valid_indexes = (
                anomaly_result.index
                .intersection(
                    context.index
                )
            )

            readable = (
                context.loc[
                    valid_indexes
                ]
                .copy()
            )

            model_result = (
                anomaly_result.loc[
                    valid_indexes
                ]
                .copy()
            )

            columns_to_join = [
                column
                for column
                in [
                    "is_anomaly",
                    "anomaly_score",
                ]
                if column
                in model_result.columns
            ]

            display = (
                readable.join(
                    model_result[
                        columns_to_join
                    ]
                )
            )

        else:

            display = (
                anomaly_result.copy()
            )

        if (
            "is_anomaly"
            in display.columns
        ):

            anomalies = (
                display[
                    display[
                        "is_anomaly"
                    ]
                    == 1
                ]
                .copy()
            )

        else:

            anomalies = (
                display.copy()
            )

        if (
            "anomaly_score"
            in anomalies.columns
        ):

            anomalies = (
                anomalies.sort_values(
                    "anomaly_score",
                    ascending=True,
                )
            )

        st.subheader(
            "Most Unusual Journeys"
        )

        useful_columns = [
            column
            for column
            in [
                "source",
                "qr_entry",
                "device",
                "country",
                "first_section",
                "last_section",
                "duration_seconds",
                "section_count",
                "unique_section_count",
                "document_open_count",
                "opened_document",
                "deep_engagement",
                "journey_score",
                "anomaly_score",
            ]
            if column
            in anomalies.columns
        ]

        with st.expander('View result records'):
            st.dataframe(
                anomalies[
                    useful_columns
                ]
                .head(50),
                width="stretch",
                hide_index=True,
            )

        return

    # =====================================================
    # CLUSTERING
    # =====================================================

    if (
        algorithm_name
        == "cluster"
    ):

        if (
            "labels"
            in result
        ):

            st.subheader(
                "Cluster Assignments"
            )

            with st.expander('View result records'):
                st.dataframe(
                    pd.DataFrame
                    .from_dict(
                        result[
                            "labels"
                        ],
                        orient="index",
                    )
                    .head(100),
                    width="stretch",
                )

        if (
            "centers"
            in result
        ):

            st.subheader(
                "Cluster Profiles"
            )

            with st.expander('View result records'):
                st.dataframe(
                    pd.DataFrame
                    .from_dict(
                        result[
                            "centers"
                        ],
                        orient="index",
                    ),
                    width="stretch",
                )

        return

    st.write(
        result
    )


# =========================================================
# HUMAN INTERPRETATION
# =========================================================

def explain_model_result(
    algorithm_name,
    result,
    target_column=None,
    context_df=None,
):

    st.subheader(
        "Human Interpretation"
    )

    if not isinstance(
        result,
        dict,
    ):

        return

    if (
        algorithm_name
        in {
            "random-forest-classification",
            "logistic-regression",
        }
    ):

        accuracy = (
            result.get(
                "accuracy"
            )
        )

        precision = (
            result.get(
                "precision"
            )
        )

        recall = (
            result.get(
                "recall"
            )
        )

        f1 = (
            result.get(
                "f1"
            )
        )

        target_name = (
            str(
                target_column
            )
            .replace(
                "_",
                " ",
            )
            if target_column
            else "target outcome"
        )

        parts = [
            (
                f"This model is trying to predict "
                f"<b>{target_name}</b>."
            )
        ]

        if accuracy is not None:

            parts.append(
                (
                    f"It correctly classified approximately "
                    f"{accuracy:.1%} of validation cases."
                )
            )

        if precision is not None:

            parts.append(
                (
                    f"Precision was {precision:.1%}."
                )
            )

        if recall is not None:

            parts.append(
                (
                    f"Recall was {recall:.1%}."
                )
            )

        if f1 is not None:

            parts.append(
                (
                    f"The F1 score was {f1:.1%}."
                )
            )

        classification_type = (
            result.get(
                "classification_type"
            )
        )

        if (
            classification_type
            == "multiclass"
        ):

            parts.append(
                (
                    f"This was a multiclass problem with "
                    f"{result.get('class_count', '?')} classes, "
                    f"so precision, recall and F1 were "
                    f"{result.get('metric_average', 'weighted')}-averaged."
                )
            )

        insight(
            "Classification meaning",
            " ".join(
                parts
            ),
        )

        importance = (
            result.get(
                "feature_importance"
            )
        )

        if importance:

            top = (
                sorted(
                    importance.items(),
                    key=lambda item:
                        item[1],
                    reverse=True,
                )[:4]
            )

            names = [
                name
                .replace(
                    "_",
                    " ",
                )
                .title()

                for name, _
                in top
            ]

            insight(
                "Business takeaway",
                (
                    "The strongest predictive signals were "
                    + ", ".join(
                        names
                    )
                    + ". These variables were most useful to the "
                    "model for distinguishing the target outcome. "
                    "Feature importance indicates predictive value, "
                    "not proof of causation."
                ),
            )

        return

    if (
        algorithm_name
        in {
            "random-forest-regression",
            "linear-regression",
        }
    ):

        score = (
            result.get(
                "score"
            )
        )

        mae = (
            result.get(
                "mae"
            )
        )

        text = ""

        if score is not None:

            text += (
                f"The model explains approximately "
                f"{max(0, score):.1%} of variation in the target."
            )

        if mae is not None:

            text += (
                f" Its average absolute prediction error "
                f"is approximately {mae:.2f} target units."
            )

        insight(
            "Regression meaning",
            text,
        )

        return

    if (
        algorithm_name
        == "isolation-forest"
    ):

        total = (
            result.get(
                "total_rows",
                0,
            )
        )

        anomalies = (
            result.get(
                "anomaly_count",
                0,
            )
        )

        rate = (
            anomalies
            / total
            * 100
            if total
            else 0
        )

        insight(
            "Anomaly meaning",
            (
                f"The model identified {anomalies:,} of "
                f"{total:,} journeys ({rate:.1f}%) as unusual. "
                f"These journeys differ most strongly from the "
                f"dominant behavioural patterns. They are candidates "
                f"for investigation, not automatically errors."
            ),
        )

        return

    if (
        algorithm_name
        == "cluster"
    ):

        centers = (
            result.get(
                "centers"
            )
        )

        groups = (
            len(
                centers
            )
            if centers
            else 0
        )

        insight(
            "Cluster meaning",
            (
                f"The model created {groups} behavioural groups. "
                f"Journeys within the same group share similar "
                f"engagement characteristics."
            ),
        )


# =========================================================
# DATE HELPERS
# =========================================================

def date_to_start_ms(
    value,
):

    dt = (
        datetime(
            value.year,
            value.month,
            value.day,
            tzinfo=
                timezone.utc,
        )
    )

    return int(
        dt.timestamp()
        * 1000
    )


def date_to_end_ms(
    value,
):

    dt = (
        datetime(
            value.year,
            value.month,
            value.day,
            23,
            59,
            59,
            999000,
            tzinfo=
                timezone.utc,
        )
    )

    return int(
        dt.timestamp()
        * 1000
    )


# =========================================================
# STREAMLIT CACHE
# =========================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def cached_real_analytics(
    start_at,
    end_at,
):

    return (
        load_real_analytics(
            start_at,
            end_at,
        )
    )


# =========================================================
# SIDEBAR
# =========================================================

if (
    LOGO_PATH.exists()
):

    st.sidebar.image(
        str(
            LOGO_PATH
        ),
        width=155,
    )


st.sidebar.markdown(
    "### GMH DPP Analytics"
)


page = (
    st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Engagement",
            "User Journey & Traceability",
            "Analytics Lab",
            "Dataset Explorer",
        ],
        label_visibility=
            "collapsed",
    )
)


st.sidebar.divider()


default_end = (
    date.today()
)

default_start = (
    default_end
    - timedelta(
        days=29
    )
)


date_range = (
    st.sidebar.date_input(
        "Real-data date range",
        value=(
            default_start,
            default_end,
        ),
    )
)


if (
    isinstance(
        date_range,
        (
            list,
            tuple,
        ),
    )
    and len(
        date_range
    )
    == 2
):

    start_date = (
        date_range[0]
    )

    end_date = (
        date_range[1]
    )

else:

    start_date = (
        default_start
    )

    end_date = (
        default_end
    )


start_at = (
    date_to_start_ms(
        start_date
    )
)

end_at = (
    date_to_end_ms(
        end_date
    )
)


if (
    st.sidebar.button(
        "Refresh real data",
        use_container_width=True,
    )
):

    st.cache_data.clear()

    st.rerun()


st.sidebar.caption(
    "Organization: GMH Gruppe"
)

st.sidebar.caption(
    GMH_ORGANIZATION
)


# =========================================================
# LOAD UMAMI
# =========================================================

if page not in {'Analytics Lab', 'Dataset Explorer'}:
    try:

        with st.spinner(
            "Loading GMH analytics..."
        ):

            real_data = (
                cached_real_analytics(
                    start_at,
                    end_at,
                )
            )

    except Exception as error:

        st.error(
            "Could not load Umami analytics."
        )

        st.exception(
            error
        )

        st.stop()


    all_events = (
        real_data[
            "events"
        ]
    )

    all_sessions = (
        real_data[
            "sessions"
        ]
    )


    # =========================================================
    # GMH FILTER
    # =========================================================

    filtered_events, filtered_sessions = filter_gmh_events(
        all_events, all_sessions, GMH_ORGANIZATION,
    )
    if not filtered_events.empty:
        gmh_custom_events = filtered_events[
            filtered_events['eventName'].isin(CUSTOM_EVENTS)
        ].copy()
    else:
        gmh_custom_events = pd.DataFrame()


    # =========================================================
    # DPP FILTER
    # =========================================================

    dpp_ids = []

    if (
        not gmh_custom_events.empty
        and "dpp_id"
        in gmh_custom_events.columns
    ):

        dpp_ids = sorted(
            value
            for value
            in gmh_custom_events[
                "dpp_id"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if value.strip()
        )


    selected_dpp = (
        st.sidebar.selectbox(
            "GMH DPP",
            [
                "All GMH DPPs",
                *dpp_ids,
            ],
        )
    )


    if selected_dpp != "All GMH DPPs":
        filtered_events, filtered_sessions = filter_selected_dpp(
            all_events, all_sessions, selected_dpp, GMH_ORGANIZATION,
        )


    # =========================================================
    # REAL ANALYTICS
    # =========================================================

    company = (
        real_company_kpis(
            real_data[
                "stats"
            ],
            filtered_events,
            filtered_sessions,
        )
    )

    sections = (
        section_performance(
            filtered_events
        )
    )

    documents = (
        document_performance(
            filtered_events
        )
    )

    dpps = (
        dpp_performance(
            filtered_events
        )
    )

    journeys = (
        build_user_journeys(
            filtered_events,
            filtered_sessions,
        )
    )

    journey_metrics = (
        journey_kpis(
            journeys
        )
    )

    qr_web_analysis = (
        qr_vs_web_engagement_analysis(
            journeys
        )
    )

    top_paths = (
        top_journeys(
            journeys
        )
    )

    entry_sections, exit_sections = (
        entry_exit_sections(
            journeys
        )
    )

    conversions = (
        section_document_conversion(
            filtered_events
        )
    )

    transitions = (
        journey_transitions(
            journeys
        )
    )

# =========================================================
# HEADER
# =========================================================

show_header()


# =========================================================
# OVERVIEW
# =========================================================

from app.dashboard_views import polish, business, guided_lab

polish()
if page in {'Overview', 'Engagement', 'User Journey & Traceability'}:
    business(page, company, journeys, journey_metrics, sections, documents, dpps,
             conversions, transitions, entry_sections, exit_sections, qr_web_analysis)
elif page == 'Analytics Lab':
    guided_lab()
    with st.expander('Advanced model explorer · synthetic experiments'):
        datasets = get_available_datasets()
        dataset_label = st.selectbox('1. Choose Dataset', list(datasets.keys()))
        dataset_key = datasets[dataset_label]
        raw_df = load_dataset(dataset_key)
        st.caption('Advanced exploration uses full-dataset preparation; scores are exploratory, not production validation. Use guided experiments for train-only preprocessing.')
        preparation = prepare_dataset_for_ml(dataset_key, raw_df)
        ml_df = preparation['dataframe']
        readable_df = preparation['cleaned_dataframe']
        numeric_columns = ml_df.select_dtypes(include='number').columns.tolist()
        if not numeric_columns:
            st.warning('No numeric model-ready columns.')
            st.stop()
        model_label = st.selectbox('2. Choose Model', list(MODEL_OPTIONS.keys()))
        algorithm_name = MODEL_OPTIONS[model_label]
        available_functions = FUNCTION_OPTIONS.get(algorithm_name, [])
        available_functions = [value for value in available_functions if value != 'predict_input']
        function_name = st.selectbox('3. Choose Function', available_functions)
        supervised_models = {'linear-regression', 'logistic-regression', 'random-forest-regression', 'random-forest-classification'}
        target_column = None
        if algorithm_name in supervised_models:
            target_column = st.selectbox('4. Choose Target', numeric_columns)
        available_features = [column for column in numeric_columns if column != target_column]
        if target_column == 'opened_document':
            leakage_columns = {'document_open_count', 'documents_opened', 'deep_engagement', 'journey_score'}
            available_features = [column for column in available_features if column not in leakage_columns]
            st.info('Leakage protection enabled: document_open_count, deep_engagement and journey_score were excluded because they already contain information about document opening.')
        elif target_column == 'deep_engagement':
            leakage_columns = {'journey_score', 'duration_seconds', 'section_count', 'unique_section_count', 'document_open_count', 'opened_document'}
            available_features = [column for column in available_features if column not in leakage_columns]
        if target_column and algorithm_name in {'random-forest-classification', 'logistic-regression'}:
            class_count = ml_df[target_column].nunique()
            if class_count > 10:
                st.warning(f'`{target_column}` has {class_count} different values. A classifier will treat each value as a separate category. If this is a numeric amount such as duration or section count, a regression model is usually more meaningful.')
            elif class_count > 2:
                st.info(f'`{target_column}` is a multiclass target with {class_count} classes. Weighted multiclass precision, recall and F1 will be used.')
        selected_features = st.multiselect('Choose Input Features', available_features, default=available_features[:min(6, len(available_features))])
        config = {'random_state': 42}
        if algorithm_name in supervised_models:
            config['test_size'] = st.slider('Test Data Share', 0.1, 0.4, 0.2, 0.05)
        if algorithm_name in {'random-forest-regression', 'random-forest-classification', 'isolation-forest'}:
            config['n_estimators'] = st.slider('Number of Trees', 100, 500, 300, 50)
        if algorithm_name == 'cluster':
            config['n_cluster'] = st.slider('Clusters', 2, 10, 3)
        if algorithm_name == 'isolation-forest':
            config['contamination'] = st.slider('Expected Unusual-Journey Share', 0.01, 0.2, 0.05, 0.01)
        if st.button('Run Advanced Synthetic Analysis', type='primary', use_container_width=True):
            if not selected_features:
                st.error('Choose at least one input feature.')
                st.stop()
            with st.expander('Preparation details'):
                show_preparation_summary(preparation, selected_features)
            if algorithm_name in supervised_models:
                total_rows = len(ml_df)
                test_rows = int(round(total_rows * config.get('test_size', 0.2)))
                train_rows = total_rows - test_rows
                st.info(f'Train/test split: {train_rows:,} training rows and {test_rows:,} test rows.')
            try:
                output = run_engine(dataframe=ml_df, algorithm_name=algorithm_name, function_name=function_name, included_columns=selected_features, target_columns=[target_column] if target_column else [], config=config, function_input={})
                result = output['result']
                st.divider()
                st.subheader('Model Result')
                display_result(result, algorithm_name, context_df=readable_df)
                explain_model_result(algorithm_name, result, target_column=target_column, context_df=readable_df)
                with st.expander('AnalyticEngine setup / scaling information'):
                    st.json(output.get('setup_data', {}))
                with st.expander('Raw engine result'):
                    st.write(result)
            except Exception as error:
                st.error('The analysis could not run.')
                st.exception(error)
elif page == 'Dataset Explorer':
    st.markdown('<div class="section-kicker">Data</div>', unsafe_allow_html=True)
    st.header('Synthetic ML Dataset Explorer')
    prototype_badge()
    datasets = get_available_datasets()
    selected_dataset = st.selectbox('Dataset', list(datasets.keys()))
    dataset_key = datasets[selected_dataset]
    raw_df = load_dataset(dataset_key)
    (c1, c2, c3) = st.columns(3)
    c1.metric('Rows', len(raw_df))
    c2.metric('Columns', len(raw_df.columns))
    c3.metric('Missing Values', int(raw_df.isna().sum().sum()))
    st.subheader('Raw Synthetic Data')
    numeric = raw_df.select_dtypes(include='number').columns.tolist()
    if numeric:
        measure = st.selectbox('Inspect a numeric distribution', numeric)
        st.altair_chart(alt.Chart(raw_df).mark_bar(color=GMH_RED).encode(x=alt.X(measure + ':Q', bin=alt.Bin(maxbins=25)), y='count():Q'), use_container_width=True)
    with st.expander('View raw synthetic records'):
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
    preparation = prepare_dataset_for_ml(dataset_key, raw_df)
    with st.expander('Preview cleaning & preparation'):
        show_preparation_summary(preparation)
        st.subheader('Prepared ML Data')
        st.dataframe(preparation['dataframe'].head(100), width='stretch', hide_index=True)
