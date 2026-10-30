import plotly.express as px
import streamlit as st
from src.ui import context, chart
from src.utils.statistics import associations, tree_mean_ci, outlier_summary
from src.preprocessing.feature_engineering import NUMERIC_FEATURES
st.title('Statistical Analysis')
c = context()
level = st.radio('Observation unit', ['Tree means (recommended for inference)', 'Individual logs (descriptive)'])
columns = [x for x in NUMERIC_FEATURES if c['logs'][x].notna().sum() > 1]
if len(columns) < 2:
    st.info('At least two measured numeric variables are needed.')
    st.stop()
raw = c['logs']
d = raw.groupby('TreeID')[columns].mean() if level.startswith('Tree') else raw[columns]
st.caption('Tree means give each tree equal weight. Individual logs within a tree are dependent; log-level p-values are descriptive only. Sampling across nearby trees may also be dependent.')
st.dataframe(d.describe().T, use_container_width=True)
method = st.selectbox('Correlation matrix', ['pearson', 'spearman'])
chart(px.imshow(d.corr(method=method), zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title=method.title() + ' association matrix'))
a,b = st.columns(2)
x = a.selectbox('Variable X', columns)
y = b.selectbox('Variable Y', columns, index=1)
try:
    st.dataframe(associations(d, x, y), use_container_width=True)
except ValueError as exc:
    st.info(str(exc))
st.caption('Pearson measures linear association; Spearman measures monotonic rank association. P-values assume independent observations and are not adjusted for multiple comparisons. Statistical significance is distinct from practical importance and does not imply causation.')
a,b = st.columns(2)
with a:
    chart(px.histogram(d, x=x, title='Measurement distribution'))
with b:
    chart(px.box(d, y=x, title='Measurement spread'))
st.dataframe(outlier_summary(d, columns), use_container_width=True)
st.caption('IQR fences flag distribution extremes, not verified errors. No outlier is deleted automatically.')
try:
    ci = tree_mean_ci(raw, x)
    st.write(f"Equal-tree-weighted mean {x}: {ci['tree_weighted_mean']:.3f}; approximate 95% bootstrap interval [{ci['low_95']:.3f}, {ci['high_95']:.3f}] across {ci['trees']} trees.")
    st.caption('1,000 resamples of tree means. Assumes trees are independent and representative; spatial dependence and a single operation limit inference. This is not a prediction interval for an individual log.')
except ValueError as exc:
    st.info(str(exc))
group = st.selectbox('Descriptive group comparison', ['ProductKey', 'SpeciesGroupKey', 'CuttingReason', 'OperatorKey'])
st.dataframe(raw.groupby(group, dropna=False)[x].agg(['count', 'mean', 'median', 'std']), use_container_width=True)
chart(px.box(raw, x=group, y=x, title='Recorded measurement by group'))
st.caption('Group differences may reflect tree size, product mix and repeated observations. This comparison is descriptive and does not estimate a causal effect.')
