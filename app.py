import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Library Deduplication Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data_from_files(original_file, unfiltered_file, review_file):
    original = pd.read_csv(original_file)
    unfiltered = pd.read_csv(unfiltered_file)
    review_queue = pd.read_csv(review_file)
    return original, unfiltered, review_queue

# Check if files exist locally (for local development)
def check_local_files():
    files = [
        'combined_library_holdings.csv',
        'splink_predictions_unfiltered.csv',
        'splink_review_queue.csv'
    ]
    return all(os.path.exists(f) for f in files)

# File upload or local load
if check_local_files():
    # Load from local files (development mode)
    original_df, unfiltered_df, review_df = load_data_from_files(
        'combined_library_holdings.csv',
        'splink_predictions_unfiltered.csv',
        'splink_review_queue.csv'
    )
    data_loaded = True
else:
    # Streamlit Cloud mode - require file upload
    st.title("📚 Library Holdings Deduplication Dashboard")
    st.markdown("### Upload Your Data Files")
    
    st.info("Please upload the three CSV files to begin analysis.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        original_file = st.file_uploader("Original Dataset (combined_library_holdings.csv)", type=['csv'])
    
    with col2:
        unfiltered_file = st.file_uploader("Unfiltered Predictions (splink_predictions_unfiltered.csv)", type=['csv'])
    
    with col3:
        review_file = st.file_uploader("Review Queue (splink_review_queue.csv)", type=['csv'])
    
    if original_file and unfiltered_file and review_file:
        original_df, unfiltered_df, review_df = load_data_from_files(
            original_file,
            unfiltered_file,
            review_file
        )
        data_loaded = True
    else:
        data_loaded = False
        st.warning("⏳ Waiting for all three files to be uploaded...")
        st.stop()

# Sidebar (only show if data is loaded)
if data_loaded:
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/books.png", width=100)
        st.title("📚 Navigation")
        st.markdown("---")
        
        st.metric("Total Records", f"{len(original_df):,}")
        st.metric("Initial Matches", f"{len(unfiltered_df):,}")
        st.metric("Filtered Matches", f"{len(review_df):,}")
        
        merge_count = len(review_df[review_df['decision'] == 'merge'])
        ignore_count = len(review_df[review_df['decision'] == 'ignore'])
        
        st.markdown("---")
        st.markdown("### Final Decisions")
        st.success(f"✅ Merge: {merge_count}")
        st.warning(f"⏭️ Ignore: {ignore_count}")
        
        st.markdown("---")
        st.markdown("""
        ### About
        This dashboard presents the library holdings deduplication analysis using 
        probabilistic record linkage (Splink) with ML-based filtering.
        
        **Process Flow:**
        1. Original Dataset
        2. Initial Matching (50% threshold)
        3. Quality Filters Applied
        4. Manual Review & Decisions
        """)

    # Main content
    st.title("📚 Library Holdings Deduplication Dashboard")
    st.markdown("### Probabilistic Record Linkage Analysis")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Overview", 
        "📊 Original Dataset", 
        "🔍 Initial Predictions", 
        "✅ Filtered Results",
        "📈 Decision Analysis",
        "🔎 Record Explorer"
    ])

    # TAB 1: Overview
    with tab1:
        st.header("Analysis Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📚 Total Records",
                value=f"{len(original_df):,}",
                help="Total library holdings in dataset"
            )
        
        with col2:
            st.metric(
                label="🔍 Initial Matches",
                value=f"{len(unfiltered_df):,}",
                help="Matches found at 50% threshold (before filtering)"
            )
        
        with col3:
            reduction_rate = ((len(unfiltered_df) - len(review_df)) / len(unfiltered_df) * 100)
            st.metric(
                label="✅ Filtered Matches",
                value=f"{len(review_df):,}",
                delta=f"-{reduction_rate:.1f}%",
                help="Matches after quality control filters"
            )
        
        with col4:
            potential_duplicates = merge_count * 2
            dedup_rate = (potential_duplicates / len(original_df) * 100)
            st.metric(
                label="🎯 Duplicates to Merge",
                value=f"{merge_count:,}",
                delta=f"{dedup_rate:.2f}% of dataset",
                help="Final confirmed duplicates"
            )
        
        st.markdown("---")
        
        # Process flow visualization
        st.subheader("🔄 Data Processing Journey")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Funnel chart
            funnel_data = {
                'Stage': ['Original Dataset', 'Initial Predictions', 'After Filters', 'To Merge', 'Final Dataset'],
                'Count': [
                    len(original_df),
                    len(unfiltered_df),
                    len(review_df),
                    merge_count,
                    len(original_df) - merge_count
                ],
                'Type': ['Input', 'Prediction', 'Filtered', 'Decision', 'Output']
            }
            
            fig = go.Figure(go.Funnel(
                y=funnel_data['Stage'],
                x=funnel_data['Count'],
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]},
            ))
            
            fig.update_layout(
                title="Record Flow Through Pipeline",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Key Metrics")
            
            filter_effectiveness = ((len(unfiltered_df) - len(review_df)) / len(unfiltered_df) * 100)
            st.metric("Filter Effectiveness", f"{filter_effectiveness:.1f}%", 
                     help="Percentage of false positives removed by filters")
            
            precision = (merge_count / len(review_df) * 100) if len(review_df) > 0 else 0
            st.metric("Merge Rate", f"{precision:.1f}%",
                     help="Percentage of filtered matches confirmed as duplicates")
            
            dup_density = (len(unfiltered_df) / len(original_df) * 100)
            st.metric("Initial Duplicate Density", f"{dup_density:.2f}%",
                     help="Percentage of records with potential duplicates")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔬 Methodology Highlights
            
            **Training Foundation:**
            - 236 manually-reviewed record pairs
            - Expectation-Maximization algorithm
            - Fellegi-Sunter probabilistic linkage
            
            **Matching Strategy:**
            - Initial threshold: 50% (high recall)
            - 6 blocking rules for comprehensive coverage
            - 4 weighted comparison fields
            """)
        
        with col2:
            st.markdown("""
            ### ✅ Quality Control Filters
            
            **5 Sequential Filters:**
            1. Different publication years (serials)
            2. Title similarity threshold (≥50%)
            3. Year range mismatches
            4. Number sequence differences
            5. Single year mismatches
            
            **Result:** High precision duplicate detection
            """)

    # TAB 2: Original Dataset
    with tab2:
        st.header("📊 Original Dataset Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_counts = original_df['SOURCE_DATASET'].value_counts()
            fig = px.bar(
                x=source_counts.index,
                y=source_counts.values,
                title="Records by Source Dataset",
                labels={'x': 'Source', 'y': 'Count'},
                color=source_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            pub_years = original_df['BEGIN_PUBLICATION_DATE'].dropna()
            fig = px.histogram(
                pub_years,
                title="Publication Year Distribution",
                labels={'value': 'Year', 'count': 'Records'},
                nbins=50,
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Data Quality Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        key_fields = ['TITLE_AUTHOR_DATE_COMBINED_NORMALIZED', 'BEGIN_PUBLICATION_DATE', 'EXTENT', 'EDITION']
        
        for col, field in zip([col1, col2, col3, col4], key_fields):
            completeness = (1 - original_df[field].isna().sum() / len(original_df)) * 100
            with col:
                st.metric(
                    label=field.replace('_', ' ').title()[:20] + "...",
                    value=f"{completeness:.1f}%",
                    help=f"Data completeness for {field}"
                )
        
        st.markdown("---")
        
        st.subheader("Sample Records")
        display_cols = ['SOURCE_DATASET', 'TITLE', 'AUTHOR', 'BEGIN_PUBLICATION_DATE', 'EDITION', 'EXTENT']
        st.dataframe(
            original_df[display_cols].head(20),
            use_container_width=True,
            height=400
        )

    # TAB 3: Initial Predictions
    with tab3:
        st.header("🔍 Initial Predictions Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_prob = unfiltered_df['match_probability'].mean()
            st.metric("Average Match Probability", f"{avg_prob:.1%}")
        
        with col2:
            high_conf = (unfiltered_df['match_probability'] >= 0.9).sum()
            st.metric("High Confidence (≥90%)", f"{high_conf:,}")
        
        with col3:
            low_conf = (unfiltered_df['match_probability'] < 0.7).sum()
            st.metric("Low Confidence (<70%)", f"{low_conf:,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                unfiltered_df,
                x='match_probability',
                title="Match Probability Distribution",
                labels={'match_probability': 'Match Probability', 'count': 'Number of Pairs'},
                nbins=50,
                color_discrete_sequence=['#EF553B']
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="green", 
                         annotation_text="Threshold (50%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            source_probs = unfiltered_df.groupby('SOURCE_DATASET_l')['match_probability'].mean().sort_values()
            fig = px.bar(
                x=source_probs.values,
                y=source_probs.index,
                orientation='h',
                title="Average Match Probability by Source",
                labels={'x': 'Avg Probability', 'y': 'Source'},
                color=source_probs.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Blocking Strategy Analysis")
        
        match_key_counts = unfiltered_df['match_key'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            **6 Blocking Rules:**
            - 0: Same source + first 6 chars
            - 1: Same source + year + first 4 chars
            - 2: Same source + exact extent
            - 3: Same source + first 10 chars
            - 4: Same source + author + year
            - 5: Same source + similar extent
            """)
            
            st.dataframe(
                pd.DataFrame({
                    'Block Rule': match_key_counts.index,
                    'Matches': match_key_counts.values,
                    'Percentage': (match_key_counts.values / len(unfiltered_df) * 100).round(1)
                }),
                hide_index=True,
                use_container_width=True
            )
        
        with col2:
            fig = px.pie(
                values=match_key_counts.values,
                names=[f"Rule {i}" for i in match_key_counts.index],
                title="Match Distribution by Blocking Rule",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Filtered Results
    with tab4:
        st.header("✅ Filtered Results Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_rate = ((len(unfiltered_df) - len(review_df)) / len(unfiltered_df) * 100)
            st.metric(
                "Records Filtered Out",
                f"{len(unfiltered_df) - len(review_df):,}",
                delta=f"{filter_rate:.1f}%"
            )
        
        with col2:
            avg_title_sim = review_df['title_similarity'].mean()
            st.metric("Avg Title Similarity", f"{avg_title_sim:.1%}")
        
        with col3:
            has_author_sim = review_df['author_similarity'].notna().sum()
            st.metric("Pairs with Author Data", f"{has_author_sim:,}")
        
        st.markdown("---")
        
        st.subheader("Filter Effectiveness")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Before Filters',
                x=['Total Predictions'],
                y=[len(unfiltered_df)],
                marker_color='#EF553B'
            ))
            
            fig.add_trace(go.Bar(
                name='After Filters',
                x=['Total Predictions'],
                y=[len(review_df)],
                marker_color='#00CC96'
            ))
            
            fig.update_layout(
                title="Impact of Quality Control Filters",
                barmode='group',
                yaxis_title="Number of Match Pairs"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                review_df,
                x='title_similarity',
                title="Title Similarity Distribution (Filtered Set)",
                labels={'title_similarity': 'Title Similarity', 'count': 'Count'},
                nbins=30,
                color_discrete_sequence=['#00CC96']
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                         annotation_text="Min Threshold (50%)")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Filtered Matches by Source")
        
        source_counts = review_df['SOURCE_DATASET_l'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                pd.DataFrame({
                    'Source': source_counts.index,
                    'Matches': source_counts.values
                }),
                hide_index=True,
                use_container_width=True,
                height=400
            )
        
        with col2:
            fig = px.bar(
                x=source_counts.index,
                y=source_counts.values,
                title="Filtered Matches by Source Dataset",
                labels={'x': 'Source', 'y': 'Match Count'},
                color=source_counts.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # TAB 5: Decision Analysis
    with tab5:
        st.header("📈 Decision Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        decision_counts = review_df['decision'].value_counts()
        merge_count = decision_counts.get('merge', 0)
        ignore_count = decision_counts.get('ignore', 0)
        
        with col1:
            st.metric("✅ Merge Decisions", f"{merge_count:,}")
        
        with col2:
            st.metric("⏭️ Ignore Decisions", f"{ignore_count:,}")
        
        with col3:
            merge_rate = (merge_count / len(review_df) * 100) if len(review_df) > 0 else 0
            st.metric("Merge Rate", f"{merge_rate:.1f}%")
        
        with col4:
            records_removed = merge_count
            final_count = len(original_df) - records_removed
            st.metric("Final Record Count", f"{final_count:,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=decision_counts.values,
                names=decision_counts.index,
                title="Decision Distribution",
                color_discrete_map={'merge': '#00CC96', 'ignore': '#FFA15A'},
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            decision_by_source = review_df.groupby(['SOURCE_DATASET_l', 'decision']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            
            if 'merge' in decision_by_source.columns:
                fig.add_trace(go.Bar(
                    name='Merge',
                    x=decision_by_source.index,
                    y=decision_by_source['merge'],
                    marker_color='#00CC96'
                ))
            
            if 'ignore' in decision_by_source.columns:
                fig.add_trace(go.Bar(
                    name='Ignore',
                    x=decision_by_source.index,
                    y=decision_by_source['ignore'],
                    marker_color='#FFA15A'
                ))
            
            fig.update_layout(
                title="Decisions by Source Dataset",
                barmode='stack',
                xaxis_title="Source",
                yaxis_title="Count",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Decision Quality Metrics")
        
        merge_df = review_df[review_df['decision'] == 'merge']
        ignore_df = review_df[review_df['decision'] == 'ignore']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=merge_df['title_similarity'],
                name='Merge',
                marker_color='#00CC96'
            ))
            
            fig.add_trace(go.Box(
                y=ignore_df['title_similarity'],
                name='Ignore',
                marker_color='#FFA15A'
            ))
            
            fig.update_layout(
                title="Title Similarity by Decision",
                yaxis_title="Title Similarity",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            merge_author = merge_df['author_similarity'].dropna()
            ignore_author = ignore_df['author_similarity'].dropna()
            
            fig = go.Figure()
            
            if len(merge_author) > 0:
                fig.add_trace(go.Box(
                    y=merge_author,
                    name='Merge',
                    marker_color='#00CC96'
                ))
            
            if len(ignore_author) > 0:
                fig.add_trace(go.Box(
                    y=ignore_author,
                    name='Ignore',
                    marker_color='#FFA15A'
                ))
            
            fig.update_layout(
                title="Author Similarity by Decision (Where Available)",
                yaxis_title="Author Similarity",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # TAB 6: Record Explorer
    with tab6:
        st.header("🔎 Record Explorer")
        
        st.markdown("### Explore Individual Match Pairs")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            decision_filter = st.selectbox(
                "Filter by Decision",
                ['All', 'merge', 'ignore']
            )
        
        with col2:
            source_filter = st.selectbox(
                "Filter by Source",
                ['All'] + sorted(review_df['SOURCE_DATASET_l'].unique().tolist())
            )
        
        with col3:
            title_sim_min = st.slider(
                "Minimum Title Similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05
            )
        
        filtered_review = review_df.copy()
        
        if decision_filter != 'All':
            filtered_review = filtered_review[filtered_review['decision'] == decision_filter]
        
        if source_filter != 'All':
            filtered_review = filtered_review[filtered_review['SOURCE_DATASET_l'] == source_filter]
        
        filtered_review = filtered_review[filtered_review['title_similarity'] >= title_sim_min]
        
        st.info(f"Showing {len(filtered_review):,} match pairs")
        
        st.markdown("---")
        
        if len(filtered_review) > 0:
            selected_idx = st.selectbox(
                "Select a match pair to view details:",
                range(len(filtered_review)),
                format_func=lambda x: f"Pair {x+1}: {filtered_review.iloc[x]['TITLE_AUTHOR_DATE_COMBINED_NORMALIZED_l'][:50]}..."
            )
            
            selected_row = filtered_review.iloc[selected_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📄 Record A")
                st.markdown(f"**ID:** `{selected_row['unique_id_l']}`")
                st.markdown(f"**Title (Normalized):** {selected_row['TITLE_AUTHOR_DATE_COMBINED_NORMALIZED_l']}")
                st.markdown(f"**Author:** {selected_row['AUTHOR_l']}")
                st.markdown(f"**Year:** {selected_row['BEGIN_PUBLICATION_DATE_l']}")
                st.markdown(f"**Edition:** {selected_row['EDITION_l']}")
                st.markdown(f"**Extent:** {selected_row['extent_normalized_l']}")
            
            with col2:
                st.markdown("### 📄 Record B")
                st.markdown(f"**ID:** `{selected_row['unique_id_r']}`")
                st.markdown(f"**Title (Normalized):** {selected_row['TITLE_AUTHOR_DATE_COMBINED_NORMALIZED_r']}")
                st.markdown(f"**Author:** {selected_row['AUTHOR_r']}")
                st.markdown(f"**Year:** {selected_row['BEGIN_PUBLICATION_DATE_r']}")
                st.markdown(f"**Edition:** {selected_row['EDITION_r']}")
                st.markdown(f"**Extent:** {selected_row['extent_normalized_r']}")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Title Similarity", f"{selected_row['title_similarity']:.2%}")
            
            with col2:
                if pd.notna(selected_row['author_similarity']):
                    st.metric("Author Similarity", f"{selected_row['author_similarity']:.2%}")
                else:
                    st.metric("Author Similarity", "N/A")
            
            with col3:
                decision_color = "🟢" if selected_row['decision'] == 'merge' else "🟡"
                st.metric("Decision", f"{decision_color} {selected_row['decision'].upper()}")
            
            st.markdown("---")
            
            st.subheader("All Filtered Match Pairs")
            
            display_cols = [
                'SOURCE_DATASET_l',
                'TITLE_AUTHOR_DATE_COMBINED_NORMALIZED_l',
                'TITLE_AUTHOR_DATE_COMBINED_NORMALIZED_r',
                'title_similarity',
                'BEGIN_PUBLICATION_DATE_l',
                'BEGIN_PUBLICATION_DATE_r',
                'decision'
            ]
            
            st.dataframe(
                filtered_review[display_cols],
                use_container_width=True,
                height=400
            )
        else:
            st.warning("No records match the current filters.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Library Holdings Deduplication Analysis</strong></p>
        <p>Methodology: Fellegi-Sunter Probabilistic Record Linkage with Sequential Quality Filtering</p>
        <p>Training Data: 236 manually-reviewed pairs | Dataset: 47,161 records</p>
    </div>
    """, unsafe_allow_html=True)