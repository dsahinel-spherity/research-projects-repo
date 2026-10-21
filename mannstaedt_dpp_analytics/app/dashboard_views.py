"""Chart-first business views; underlying records remain available on demand."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

RED = '#E30613'
PALETTE = [RED, '#8C1821', '#E76D75', '#A6A8AF', '#656975', '#F3B4B9', '#454953', '#C74853', '#7E8793', '#D3D5DB', '#333640']


def polish():
    st.markdown('''<style>
    .block-container {max-width:1480px;padding-top:1.6rem;}
    h1,h2,h3 {letter-spacing:-.025em;}
    [data-testid="stMetric"] {border:1px solid #292B30;border-top:3px solid #E30613;border-radius:12px;background:#18191D;}
    [data-testid="stExpander"] {border:1px solid #292B30;border-radius:12px;}
    .stButton>button {border-radius:9px;}
    </style>''', unsafe_allow_html=True)


def table(data, label='Explore all records'):
    with st.expander(label):
        st.dataframe(data, use_container_width=True, hide_index=True)


def chart_style(chart):
    return chart.configure_view(strokeOpacity=0).configure_axis(gridColor='#292B30', labelColor='#A6A8AF', titleColor='#A6A8AF').configure(background='#18191D')


def ranked(data, category, value, title, key, ascending=False):
    st.subheader(title)
    if data is None or data.empty:
        st.info('No matching activity in this period.')
        return
    ordered = data.sort_values(value, ascending=ascending)
    show_all = st.checkbox(f'Show all {len(data):,} categories', key=f'all_{key}') if len(data) > 10 else False
    shown = ordered if show_all else ordered.head(10)
    st.caption(f'{len(shown):,} of {len(data):,} categories · hover for exact values')
    chart = alt.Chart(shown).mark_bar(color=RED, cornerRadiusEnd=4).encode(
        y=alt.Y(f'{category}:N', sort=shown[category].tolist(), title=None, axis=alt.Axis(labelLimit=330)),
        x=alt.X(f'{value}:Q', title=value.replace('_', ' ').title()), tooltip=[alt.Tooltip(c) for c in data.columns if not data[c].map(lambda v: isinstance(v, (list, dict))).any()])
    st.altair_chart(chart_style(chart.properties(height=max(180, len(shown)*30))), use_container_width=True)
    table(data, f'View all data · {title}')


def donut(data, category, value, title):
    st.subheader(title)
    if data.empty:
        st.info('No matching journeys.')
        return
    data = data.sort_values(value, ascending=False).copy()
    if len(data) > 10:
        data = pd.concat([data.head(10), pd.DataFrame([{category: 'Other', value: data.iloc[10:][value].sum()}])], ignore_index=True)
    chart = alt.Chart(data).mark_arc(innerRadius=65, outerRadius=105).encode(theta=alt.Theta(f'{value}:Q'), color=alt.Color(f'{category}:N', scale=alt.Scale(range=PALETTE), legend=alt.Legend(title=None)), tooltip=[category, value])
    st.altair_chart(chart_style(chart.properties(height=270)), use_container_width=True)
    table(data, f'View chart data · {title}')


def distribution(journeys, field):
    return journeys[field].fillna('Unknown').replace('', 'Unknown').value_counts().rename_axis(field).reset_index(name='journeys')


def behavior(journeys):
    st.subheader('Behavior signals')
    st.caption('Descriptive signals within tracked journeys, not evidence of frustration or causation. Pageview-only visits are excluded.')
    if journeys.empty:
        st.info('Tracked journeys are needed to identify behavior signals.')
        return
    total = len(journeys)
    signals = [
        ('Exploration without a document', (journeys.section_count >= 3) & ~journeys.opened_document, 'Check whether document links are visible from the most explored sections. These users may also have found everything they needed without a document.'),
        ('Single-section journeys', (journeys.section_count == 1) & ~journeys.opened_document, 'Review first-section clarity and onward links. A short visit may be successful, so validate with a usability test.'),
        ('Repeated section navigation', journeys.section_count > journeys.unique_section_count, 'Inspect repeated paths for unclear navigation or comparison behavior. Consider clearer section names and navigation breadcrumbs.'),
        ('Long journeys without a document', (journeys.duration_seconds >= 120) & ~journeys.opened_document, 'Review document discoverability and content clarity. Elapsed time also includes inactivity.'),
    ]
    for title, mask, action in signals:
        with st.container(border=True):
            st.markdown(f'**{title}**')
            st.metric('Journeys matching this signal', f'{int(mask.sum()):,}', f'{mask.mean()*100:.1f}% of {total:,}', delta_color='off')
            st.write(action)
            with st.expander('Inspect matching journeys'):
                records = journeys.loc[mask].copy()
                columns = [c for c in ['journey_id', 'dpp_id', 'dpp_url', 'dpp_path', 'source', 'section_count', 'duration_seconds', 'journey_path'] if c in records.columns]
                st.dataframe(
                    records[columns], use_container_width=True, hide_index=True,
                    column_config={
                        'dpp_id': st.column_config.TextColumn('DPP identifier'),
                        'dpp_url': st.column_config.LinkColumn('DPP URL', display_text='Open DPP'),
                        'dpp_path': st.column_config.TextColumn('Recorded DPP path'),
                    },
                )
                if not records.empty and 'dpp_url' in records and records.dpp_url.eq('').any():
                    st.caption('Some records lack a usable DPP path. Their identifier is shown, but a URL cannot be reconstructed from the recorded event.')
                if 'dpp_count' in records and records.dpp_count.gt(1).any():
                    st.caption('Some journeys visit multiple DPPs. The identifier and link refer to the first recorded DPP; behavior metrics cover the entire journey.')
    st.caption('Signals overlap. Thresholds are review heuristics, not validated alert limits.')


def business(page, company, journeys, metrics, sections, documents, dpps, conversions, transitions, entries, exits, comparison):
    st.caption('REAL UMAMI DATA · GMH GRUPPE')
    st.title({'Overview':'DPP performance at a glance', 'Engagement':'What visitors explore', 'User Journey & Traceability':'From entry to document discovery'}[page])
    st.caption('GMH-associated sessions · selected period and DPP · journey breaks after more than 10 minutes of inactivity')
    cols = st.columns(4)
    for col, label, value in zip(cols, ['Pageviews','GMH sessions','Tracked journeys','Document-open rate'], [f"{company['pageviews']:,}", f"{company['visits']:,}", f'{len(journeys):,}', f"{metrics['document_conversion_percent']:.1f}%"]):
        col.metric(label, value)
    cols = st.columns(4)
    for col, label, value in zip(cols, ['Section views','Document opens','Sections per journey','Observed journey duration'], [company['section_views'],company['document_opens'],metrics['avg_sections'],f"{metrics['avg_duration_seconds']:.0f}s"]):
        col.metric(label,value)
    if len(journeys) < 20:
        st.info('Small sample: use these patterns to choose what to investigate, not to draw firm conclusions.')
    if page == 'Overview':
        if not journeys.empty:
            st.subheader('Journey activity over time')
            daily = journeys.assign(day=journeys.started_at.dt.floor('D')).groupby('day', as_index=False).agg(Journeys=('journey_id','size'), Documents=('document_open_count','sum'))
            trend = daily.melt('day', var_name='Measure', value_name='Count')
            st.altair_chart(chart_style(alt.Chart(trend).mark_line(point=True).encode(x=alt.X('day:T',title=None),y='Count:Q',color=alt.Color('Measure:N',scale=alt.Scale(range=[RED,'#A6A8AF'])),tooltip=['day:T','Measure:N','Count:Q']).properties(height=260)), use_container_width=True)
            table(daily, 'View daily counts')
            a,b=st.columns(2)
            with a: donut(distribution(journeys,'source'),'source','journeys','QR / web journey mix')
            with b:
                outcome = pd.DataFrame({'Outcome':['Opened a document','No document recorded'], 'Journeys':[int(journeys.opened_document.sum()),int((~journeys.opened_document).sum())]})
                donut(outcome,'Outcome','Journeys','Document discovery')
        a,b=st.columns(2)
        with a: ranked(sections,'section','views','Most explored sections','overview_sections')
        with b: ranked(documents,'document','opens','Most opened documents','overview_docs')
        st.subheader('Do QR and web journeys behave differently?')
        st.write(comparison['conclusion'])
        comp=comparison['table']
        if not comp.empty:
            measure=st.selectbox('Compare', ['Document Conversion (%)','Avg Sections','Avg Duration (sec)'])
            ranked(comp,'Source',measure,'QR vs web comparison','qr_compare')
        behavior(journeys)
    elif page == 'Engagement':
        ranked(sections,'section','views','Section demand','eng_sections')
        ranked(documents,'document','opens','Document demand','eng_docs')
        ranked(dpps,'dpp_id','engagement_events','DPP engagement','eng_dpps')
        if not journeys.empty:
            a,b=st.columns(2)
            with a: donut(distribution(journeys,'device'),'device','journeys','Device mix')
            with b: ranked(distribution(journeys,'country'),'country','journeys','Countries','countries')
            st.subheader('Where is document discovery weaker?')
            dimension=st.selectbox('Compare journey groups', ['device','source','country'])
            minimum=st.slider('Minimum journeys per group',1,50,5)
            groups=journeys.assign(**{dimension: journeys[dimension].fillna('Unknown').replace('','Unknown')}).groupby(dimension,as_index=False).agg(journeys=('journey_id','size'), document_rate=('opened_document','mean'))
            groups['document_rate']=groups.document_rate*100
            ranked(groups[groups.journeys>=minimum],dimension,'document_rate','Document-open rate (%)','segment',ascending=True)
            st.caption('Compare sample sizes before prioritizing. Device or country differences do not establish why visitors behave differently.')
    else:
        if not journeys.empty:
            paths=journeys.journey_path.value_counts().rename_axis('path').reset_index(name='journeys')
            ranked(paths,'path','journeys','Most common paths','paths')
        a,b=st.columns(2)
        with a: ranked(entries,'section','journeys','First sections','entry')
        with b: ranked(exits,'section','journeys','Last observed sections','exit')
        st.caption('The last observed section is a proxy for exit; the final dwell time and actual exit are not recorded.')
        ranked(conversions,'section','document_conversion_percent','Sections followed by a document (%)','conversion')
        st.caption('Denominator: section-view occurrences. A later document can count for several preceding views in the same journey.')
        if not transitions.empty:
            transitions=transitions.assign(transition=transitions['from']+' → '+transitions['to'])
        ranked(transitions,'transition','journeys','Common transitions','transitions')
        table(journeys,'Explore individual journeys and metadata')
        behavior(journeys)


@st.cache_data(ttl=600, show_spinner=False)
def experiment(question, model, share, contamination):
    """Cache reports including the fitted cleaning and filtering audit."""
    from app.services.lab_service import run_experiment
    return run_experiment(question, model, share, contamination)


def cleaning_details(report):
    with st.expander('Cleaning and filtering · what happened before the results?', expanded=False):
        if not report:
            st.info('Run the experiment again to see cleaning details for this report.')
            return
        st.markdown('**1. Clean the synthetic journey records**')
        st.write(
            f"Started with {report['original_rows']:,} rows. Removed "
            f"{report['duplicates_removed']:,} exact duplicates and "
            f"{report['missing_target_rows_removed']:,} rows with a missing outcome. "
            f"Retained {report['retained_rows']:,} rows."
        )
        st.caption('Only duplicate records and missing outcomes are removed; no GMH, date, device or country filter is applied to this synthetic dataset.')
        st.markdown('**2. Select the inputs for this question**')
        st.write('Outcome to predict:', report['target'] or 'None — unusual-journey detection is unsupervised.')
        st.write('Included model inputs:', ', '.join(report['inputs']))
        st.write('Excluded from model inputs:', ', '.join(report['excluded_inputs']) or 'None')
        st.caption('Excluded fields remain in the source dataset. Supervised input selection avoids outcome counts and derived scores that could reveal the answer.')
        st.markdown('**3. Fill missing inputs and transform the data**')
        st.write(f"Filled {report['missing_values_filled']:,} missing input values across the retained rows. Numeric inputs use the fitted median; categorical inputs use the fitted most frequent value.")
        st.dataframe(pd.DataFrame(report['replacements']), use_container_width=True, hide_index=True)
        st.write(f"Categories become indicator columns (one-hot encoding); numeric inputs are standardized. The model receives {report['encoded_columns']:,} transformed input columns. Unseen categories are safely ignored during encoding.")
        st.markdown('**4. Keep evaluation separate**')
        if report['evaluation_rows']:
            st.write(f"{report['fit_rows']:,} training rows and {report['evaluation_rows']:,} held-out evaluation rows. Replacement values, categories and scaling are learned only from training rows, then applied unchanged to evaluation rows.")
        else:
            st.write(f"All {report['fit_rows']:,} retained rows are used to fit preprocessing and the anomaly model. There is no held-out evaluation; the output is an in-sample review ranking.")


def guided_lab():
    from app.services.lab_service import QUESTIONS
    st.title('Analytics Lab')
    st.warning('SYNTHETIC DATA ONLY · These experiments demonstrate methods, not real GMH outcomes or production-validated predictions.')
    st.caption('Choose a question, evaluate against a simple baseline, and turn the result into a testable product hypothesis.')
    with st.form('guided_experiment'):
        question=st.selectbox('What would you like to understand?',list(QUESTIONS))
        model=st.selectbox('Model family (supervised questions)', ['Random forest','Linear / logistic baseline challenger'])
        share=st.slider('Hold out for evaluation',0.15,0.4,0.25,0.05)
        contamination=st.slider('Review budget for unusual journeys',0.01,0.15,0.05,0.01)
        run=st.form_submit_button('Run synthetic experiment',type='primary')
    if run:
        with st.spinner('Evaluating synthetic journeys…'):
            try:
                result=experiment(question,model,share,contamination)
                st.session_state['lab_report']=(question,model,result)
            except Exception as error:
                st.error(f'This experiment could not run: {error}')
    if 'lab_report' not in st.session_state:
        st.info('Start with document discovery, engagement, navigation depth, duration or unusual journeys.')
        return
    q,m,result=st.session_state['lab_report']
    st.subheader(q)
    st.caption(f'Last completed experiment · {m} · fixed reproducible seed')
    st.write(result['action'])
    cleaning_details(result.get('preparation'))
    st.subheader('Experiment results')
    if result['kind']=='anomaly':
        st.metric('Journeys selected for review',f"{result['flagged']} / {result['rows']}")
        records=result['records']
        st.altair_chart(chart_style(alt.Chart(records).mark_bar(color=RED).encode(x=alt.X('review_score:Q',bin=alt.Bin(maxbins=30),title='Unusualness score (higher = more unusual)'),y=alt.Y('count():Q',title='Journeys'),tooltip=['count():Q']).properties(height=250)),use_container_width=True)
        table(records.head(10),'Inspect top 10 unusual journeys')
        table(records,'Explore all scored journeys')
        st.caption('In-sample unsupervised ranking. Review budget controls flagged share; no labeled anomalies exist to measure detection accuracy.')
    else:
        cols=st.columns(3)
        for col,(label,value) in zip(cols,result['metrics'].items()):
            col.metric(label, f'{value:.1%}' if result['kind']=='classifier' else f'{value:.2f}')
        st.caption(f"{result['train_rows']} training journeys · {result['test_rows']} unseen evaluation journeys. Imputation, encoding and scaling are fitted only on training data.")
        metrics=result['metrics']
        improved=metrics['Balanced accuracy']>metrics['Baseline balanced accuracy'] if result['kind']=='classifier' else metrics['Mean absolute error']<metrics['Baseline error']
        st.info('This model beats the simple baseline on this synthetic split. Confirm stability and usefulness on representative data before deployment.' if improved else 'This model does not beat the simple baseline on this split. These inputs may not contain enough useful signal; do not deploy it based on this result.')
        if result['kind']=='classifier':
            st.caption('Balanced accuracy weights both classes equally. Baseline always predicts the most common training class. F1 balances finding document/engagement outcomes with avoiding false positives.')
            c=alt.Chart(result['confusion']).mark_rect().encode(x='Predicted:N',y='Actual:N',color=alt.Color('Journeys:Q',scale=alt.Scale(range=['#292B30',RED])),tooltip=['Actual','Predicted','Journeys'])
            st.altair_chart(chart_style((c+c.mark_text(color='white').encode(text='Journeys:Q')).properties(height=220)),use_container_width=True)
        else:
            unit='seconds' if result['target']=='duration_seconds' else 'sections'
            st.caption(f'Mean absolute error is measured in {unit}; lower is better. The baseline always predicts the training median. R² can be negative when predictions are poor.')
            p=result['predictions']
            scatter=alt.Chart(p).mark_circle(color=RED,opacity=.45).encode(x='Actual:Q',y='Predicted:Q',tooltip=['Actual','Predicted'])
            low=float(min(p.min()));high=float(max(p.max()))
            diagonal=alt.Chart(pd.DataFrame({'Actual':[low,high],'Predicted':[low,high]})).mark_line(color='#A6A8AF',strokeDash=[5,5]).encode(x='Actual:Q',y='Predicted:Q')
            st.altair_chart(chart_style((scatter+diagonal).properties(height=280)),use_container_width=True)
        ranked(result['importance'],'Feature','Importance','Which inputs help on held-out journeys?','importance')
        st.caption('Permutation importance measures score loss after shuffling an input. Near-zero or negative values suggest little reliable benefit. This is association, not causation.')
        table(result['predictions'],'Inspect held-out predictions')
    with st.expander('Method, inputs and limitations'):
        st.write('Inputs:',result['features'])
        st.write('Synthetic generator assumptions determine these patterns. Entry context is used for depth, duration and deep engagement; document models additionally use section exploration, so they are retrospective, not entry-time forecasts. Outcome counts and derived scores are excluded from supervised inputs. A single held-out split is not production validation.')
