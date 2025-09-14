import streamlit as st
import pandas as pd
import numpy as np
from mapping import suggest_mapping, load_history, save_history
from data_preprocessor import preprocess_dataframe, DataPreprocessor
from llm_assistant import create_llm_assistant
import io
import re
import json

CANONICAL_SCHEMA = {
    "order_id": "Unique order identifier",
    "order_date": "ISO date of order",
    "customer_id": "Internal customer id",
    "customer_name": "Full name",
    "email": "Contact email",
    "phone": "Contact phone",
    "billing_address": "Billing address line",
    "shipping_address": "Shipping address line",
    "city": "City",
    "state": "State/Province",
    "postal_code": "Zip/Postal",
    "country": "Country",
    "product_sku": "SKU code",
    "product_name": "Item name",
    "category": "Category",
    "subcategory": "Subcategory if any",
    "quantity": "Units ordered",
    "unit_price": "Price per unit",
    "currency": "Currency code",
    "discount_pct": "Discount fraction (0-1)",
    "tax_pct": "Tax fraction (0-1)",
    "shipping_fee": "Shipping amount",
    "total_amount": "Total amount charged",
    "tax_id": "Tax/GST/VAT identifier"
}

# Page configuration
st.set_page_config(
    page_title="Smart Column Mapper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'llm_assistant' not in st.session_state:
    st.session_state['llm_assistant'] = None
if 'llm_suggestions' not in st.session_state:
    st.session_state['llm_suggestions'] = None
if 'original_df' not in st.session_state:
    st.session_state['original_df'] = None
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None
if 'preprocessing_report' not in st.session_state:
    st.session_state['preprocessing_report'] = None
# Add this new session state variable to store AI-enhanced mappings
if 'ai_enhanced_mapping' not in st.session_state:
    st.session_state['ai_enhanced_mapping'] = None
if 'current_columns' not in st.session_state:
    st.session_state['current_columns'] = None

# Sidebar with help and tips
with st.sidebar:
    st.write("## 🤖 LLM Assistant Setup")

    # Google API Key input
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Enter your Google Gemini API key to enable AI-powered mapping validation"
    )

    if google_api_key and google_api_key.strip():
        if st.button("🔗 Connect LLM Assistant"):
            with st.spinner("Connecting to LLM..."):
                try:
                    llm_assistant = create_llm_assistant(
                        google_api_key.strip())
                    if llm_assistant:
                        st.session_state['llm_assistant'] = llm_assistant
                        st.success("✅ LLM Assistant connected!")
                    else:
                        st.error("❌ Failed to connect LLM Assistant")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")

    if st.session_state['llm_assistant']:
        st.success("🤖 LLM Assistant: Ready")
    elif not google_api_key or not google_api_key.strip():
        st.info("💡 Enter your Google API key above to enable AI features")

    st.write("---")
    st.write("## 💡 Tips & Help")

    with st.expander("🤖 LLM Bot Check"):
        st.write("""
        **AI-Powered Validation:**
        - Reviews rule-based suggestions
        - Analyzes actual data content
        - Suggests improvements
        - Provides confidence scores
        - Explains mapping decisions
        """)

    with st.expander("🔧 Preprocessing Benefits"):
        st.write("""
        **Data preprocessing helps with:**
        - Standardizing date formats
        - Cleaning phone numbers and emails  
        - Handling missing values intelligently
        - Removing duplicate rows
        - Validating data quality
        - Converting data types properly
        """)

    with st.expander("🎯 Mapping Confidence"):
        st.write("""
        **Confidence Indicators:**
        - 🎯 High (>80%): Very confident match
        - 🔍 Medium (50-80%): Likely match
        - ❓ Low (<50%): Uncertain match
        - 🤖 AI-enhanced suggestion
        
        **Based on:**
        - Column name similarity
        - Data pattern analysis  
        - Historical mappings
        - Data content validation
        """)

    with st.expander("📊 Data Quality"):
        st.write("""
        **Quality Checks:**
        - Missing value ratios
        - Data type consistency
        - Pattern validation
        - Duplicate detection
        - Outlier identification
        """)

    st.write("---")
    st.write("### 🚀 Quick Start")
    st.write("""
    1. Upload your CSV/XLSX file
    2. Configure preprocessing options
    3. Click 'Preprocess Data'  
    4. Review suggested mappings
    5. Use 'Bot Check' for AI validation
    6. Confirm and download results
    """)

# Main app
st.title("🧠 Smart Column Mapper with AI Assistant")

uploaded_file = st.file_uploader(
    "Upload your CSV/XLSX file", type=["csv", "xlsx"])

if uploaded_file:
    # Load input file
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ File loaded successfully! Shape: {df.shape}")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()

    st.write("### 📊 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Data preprocessing section
    st.write("### 🔧 Data Preprocessing")

    with st.expander("⚙️ Preprocessing Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            remove_duplicates = st.checkbox(
                "Remove duplicate rows", value=True)
            handle_missing = st.checkbox("Handle missing values", value=True)
            standardize_formats = st.checkbox(
                "Standardize data formats", value=True)

        with col2:
            missing_threshold = st.slider("Missing value threshold (%)", 0, 100, 80,
                                          help="Remove columns with more than this % of missing values") / 100
            validate_data = st.checkbox("Validate data quality", value=True)
            outlier_detection = st.checkbox("Detect outliers", value=False)

    # Create preprocessing config
    preprocessing_config = {
        'remove_duplicates': remove_duplicates,
        'handle_missing': handle_missing,
        'standardize_formats': standardize_formats,
        'validate_data': validate_data,
        'missing_threshold': missing_threshold,
        'outlier_detection': outlier_detection,
        'date_format': 'infer',
        'currency_symbol': '₹',
        'phone_country_code': '+91',
        'default_country': 'India'
    }

    # Preprocess the data
    if st.button("🚀 Preprocess Data"):
        with st.spinner("Preprocessing data..."):
            try:
                processed_df, preprocessing_report = preprocess_dataframe(
                    df, config=preprocessing_config
                )

                st.session_state['original_df'] = df.copy()
                st.session_state['processed_df'] = processed_df.copy()
                st.session_state['preprocessing_report'] = preprocessing_report

                # Clear AI mappings when data changes
                st.session_state['ai_enhanced_mapping'] = None
                st.session_state['llm_suggestions'] = None

                st.success("✅ Data preprocessing completed!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")

    # Use processed data if available, otherwise use original
    if st.session_state['processed_df'] is not None:
        df_to_use = st.session_state['processed_df']
        st.success("✅ Using preprocessed data for mapping")

        # Show preprocessing report
        with st.expander("📋 Preprocessing Report", expanded=False):
            report = st.session_state['preprocessing_report']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", report['original_shape'][0])
                st.metric("Final Rows", report['final_shape'][0])
            with col2:
                st.metric("Original Columns", report['original_shape'][1])
                st.metric("Final Columns", report['final_shape'][1])
            with col3:
                st.metric("Changes Made", len(report['changes']))
                st.metric("Warnings", len(report['warnings']))

            if report['changes']:
                st.write("**Changes Made:**")
                for change in report['changes']:
                    st.write(f"• {change}")

            if report['warnings']:
                st.write("**Warnings:**")
                for warning in report['warnings']:
                    st.warning(warning)

            if report['data_quality_issues']:
                st.write("**Data Quality Issues:**")
                for issue in report['data_quality_issues']:
                    st.write(f"• {issue}")

        st.write("### 📊 Preprocessed Data Preview")
        st.dataframe(df_to_use.head(), use_container_width=True)

        # Compare with original if requested
        if st.checkbox("Compare with original data"):
            st.write("**Original vs Preprocessed Comparison:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Data:**")
                st.dataframe(st.session_state['original_df'].head())
            with col2:
                st.write("**Preprocessed Data:**")
                st.dataframe(df_to_use.head())
    else:
        df_to_use = df
        st.info(
            "💡 Click 'Preprocess Data' to clean and standardize your data before mapping")

    # Load history
    history = load_history()

    # Get AI-powered suggestions with data analysis
    input_columns = list(df_to_use.columns)

    # Check if columns have changed (reset AI mappings if they have)
    if st.session_state['current_columns'] != input_columns:
        st.session_state['current_columns'] = input_columns
        st.session_state['ai_enhanced_mapping'] = None
        st.session_state['llm_suggestions'] = None

    # Use AI-enhanced mapping if available, otherwise generate fresh suggestions
    if st.session_state['ai_enhanced_mapping'] is not None:
        suggested_mapping = st.session_state['ai_enhanced_mapping']
        st.info("🤖 Using AI-enhanced suggestions")
    else:
        with st.spinner("Analyzing columns and generating suggestions..."):
            suggested_mapping = suggest_mapping(
                input_columns, df_to_use, history)

    st.write("### 🎯 Smart Column Mapping Suggestions")

    # Show confidence levels
    st.write("**Initial Mapping Confidence:**")
    confidence_data = []
    for col in input_columns:
        suggestion = suggested_mapping.get(col)
        if suggestion and suggestion.get('field'):
            confidence_data.append({
                'Column': col,
                'Suggested Mapping': suggestion['field'],
                'Confidence': f"{suggestion.get('confidence', 0):.1%}",
                'Reason': suggestion.get('reason', 'Unknown')
            })

    if confidence_data:
        st.dataframe(pd.DataFrame(confidence_data), use_container_width=True)

        # LLM Bot Check Section
        if st.session_state['llm_assistant']:
            st.write("---")
            st.write("### 🤖 AI Bot Check")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Want AI to review and improve these suggestions?**")
                st.caption(
                    "The AI will analyze your actual data content and provide enhanced suggestions")
            with col2:
                if st.button("🔍 Bot Check Mappings", type="primary"):
                    with st.spinner("🤖 AI is analyzing your mappings..."):
                        try:
                            # Get LLM validation using the current suggested_mapping (not AI-enhanced one)
                            base_mapping = suggest_mapping(
                                input_columns, df_to_use, history) if st.session_state['ai_enhanced_mapping'] is not None else suggested_mapping

                            llm_response = st.session_state['llm_assistant'].validate_and_improve_mappings(
                                original_columns=input_columns,
                                current_mappings=base_mapping,
                                sample_data=df_to_use.head(10),
                                canonical_schema=CANONICAL_SCHEMA
                            )

                            st.session_state['llm_suggestions'] = llm_response

                            if 'error' in llm_response:
                                st.error(
                                    f"❌ LLM Error: {llm_response['error']}")
                            else:
                                st.success("✅ AI analysis completed!")
                                st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error during AI analysis: {str(e)}")

        # Show LLM suggestions if available
        if st.session_state['llm_suggestions'] and 'improved_mappings' in st.session_state['llm_suggestions']:
            st.write("### 🤖 AI-Enhanced Mapping Suggestions")

            llm_data = st.session_state['llm_suggestions']
            improved_mappings = llm_data.get('improved_mappings', {})
            validation_summary = llm_data.get('validation_summary', {})
            recommendations = llm_data.get('recommendations', [])

            # Show validation summary
            if validation_summary:
                st.write("**AI Analysis Summary:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Columns", validation_summary.get(
                        'total_columns', 0))
                with col2:
                    st.metric("Approved", validation_summary.get(
                        'mappings_approved', 0))
                with col3:
                    st.metric("Changed", validation_summary.get(
                        'mappings_changed', 0))
                with col4:
                    st.metric(
                        "AI Confidence", f"{validation_summary.get('overall_confidence', 0):.1%}")

            # Show improved mappings
            if improved_mappings:
                ai_suggestions_data = []
                for col, suggestion in improved_mappings.items():
                    ai_suggestions_data.append({
                        'Column': col,
                        'AI Suggestion': suggestion.get('suggested_field', ''),
                        'Confidence': f"{suggestion.get('confidence', 0):.1%}",
                        'AI Reason': suggestion.get('reason', ''),
                        'Changes': suggestion.get('changes_from_original', 'No changes')
                    })

                if ai_suggestions_data:
                    st.dataframe(pd.DataFrame(ai_suggestions_data),
                                 use_container_width=True)

                # Option to apply AI suggestions
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🚀 Apply AI Suggestions"):
                        # Create a new mapping based on current suggested_mapping
                        base_mapping = suggest_mapping(
                            input_columns, df_to_use, history) if st.session_state['ai_enhanced_mapping'] is None else st.session_state['ai_enhanced_mapping']
                        ai_enhanced_mapping = base_mapping.copy()

                        # Update with AI improvements
                        for col, ai_suggestion in improved_mappings.items():
                            if col in ai_enhanced_mapping:
                                ai_enhanced_mapping[col].update({
                                    'field': ai_suggestion.get('suggested_field'),
                                    'confidence': ai_suggestion.get('confidence', 0),
                                    'reason': f"AI: {ai_suggestion.get('reason', '')}"
                                })
                            else:
                                # Create new mapping entry for unmapped columns
                                ai_enhanced_mapping[col] = {
                                    'field': ai_suggestion.get('suggested_field'),
                                    'confidence': ai_suggestion.get('confidence', 0),
                                    'reason': f"AI: {ai_suggestion.get('reason', '')}"
                                }

                        # Store in session state
                        st.session_state['ai_enhanced_mapping'] = ai_enhanced_mapping
                        # Clear suggestions UI
                        st.session_state['llm_suggestions'] = None

                        st.success(
                            "✅ AI suggestions applied! Please review the updated mappings below.")
                        st.rerun()

                with col2:
                    if st.button("❌ Dismiss AI Suggestions"):
                        st.session_state['llm_suggestions'] = None
                        st.rerun()

            # Show recommendations
            if recommendations:
                st.write("**🎯 AI Recommendations:**")
                for rec in recommendations:
                    st.info(f"💡 {rec}")

        else:
            if not st.session_state['llm_assistant']:
                st.info(
                    "🔗 Connect LLM Assistant in the sidebar to enable AI-powered mapping validation")

    else:
        st.info(
            "No confident mappings found. Please review manual mapping options below.")

        # Still offer LLM assistance for unmapped columns
        if st.session_state['llm_assistant']:
            if st.button("🤖 Get AI Suggestions for Unmapped Columns"):
                with st.spinner("🤖 AI is analyzing unmapped columns..."):
                    try:
                        unmapped_response = st.session_state['llm_assistant'].get_field_suggestions_for_unmapped(
                            unmapped_columns=input_columns,
                            sample_data=df_to_use.head(10),
                            canonical_schema=CANONICAL_SCHEMA
                        )

                        if 'suggestions' in unmapped_response and unmapped_response['suggestions']:
                            # Create new AI-enhanced mapping
                            ai_enhanced_mapping = suggested_mapping.copy()

                            # Update with AI suggestions
                            for col, ai_suggestion in unmapped_response['suggestions'].items():
                                ai_enhanced_mapping[col] = {
                                    'field': ai_suggestion.get('suggested_field'),
                                    'confidence': ai_suggestion.get('confidence', 0),
                                    'reason': f"AI: {ai_suggestion.get('reason', '')}"
                                }

                            # Store in session state
                            st.session_state['ai_enhanced_mapping'] = ai_enhanced_mapping

                            st.success("✅ AI suggestions added!")
                            st.rerun()
                        else:
                            st.warning(
                                "AI couldn't find confident suggestions for unmapped columns")

                    except Exception as e:
                        st.error(f"Error getting AI suggestions: {str(e)}")

    # Manual mapping configuration
    new_mapping = {}
    st.write("### ⚙️ Configure Final Mappings")
    st.caption(
        "Review and adjust the mappings below. AI-enhanced suggestions are marked with 🤖")

    for col in input_columns:
        suggestion = suggested_mapping.get(col, {})
        default_field = suggestion.get('field')
        confidence = suggestion.get('confidence', 0)
        reason = suggestion.get('reason', '')

        # Create label with confidence indicator
        label = f"Map '{col}' →"
        if confidence > 0:
            confidence_emoji = "🎯" if confidence > 0.8 else "🔍" if confidence > 0.5 else "❓"
            label += f" {confidence_emoji} ({confidence:.0%})"

        # Show AI badge if this suggestion came from AI
        if reason.startswith("AI:"):
            label += " 🤖"

        # Column layout for mapping selection and explanation
        col_main, col_explain = st.columns([3, 1])

        with col_main:
            chosen = st.selectbox(
                label,
                options=["(ignore)"] + list(CANONICAL_SCHEMA.keys()),
                index=(list(CANONICAL_SCHEMA.keys()).index(
                    default_field) + 1) if default_field in CANONICAL_SCHEMA else 0,
                help=f"Suggestion reason: {reason}" if reason else None,
                key=f"mapping_{col}"
            )

        with col_explain:
            # Show AI explanation button if LLM is available and field is mapped
            if st.session_state['llm_assistant'] and chosen != "(ignore)" and chosen in CANONICAL_SCHEMA:
                if st.button("🤖 Explain", key=f"explain_{col}", help=f"Get AI explanation for mapping '{col}' → '{chosen}'"):
                    with st.spinner("Getting AI explanation..."):
                        try:
                            explanation = st.session_state['llm_assistant'].explain_mapping_choice(
                                column_name=col,
                                suggested_field=chosen,
                                sample_values=df_to_use[col].dropna().head(
                                    3).tolist(),
                                canonical_schema=CANONICAL_SCHEMA
                            )
                            st.info(
                                f"🤖 **AI Explanation for '{col}' → '{chosen}':** {explanation}")
                        except Exception as e:
                            st.error(f"Error getting explanation: {str(e)}")

        if chosen != "(ignore)":
            new_mapping[col] = chosen

    # Check for potential duplicate mappings and warn user
    if new_mapping:
        reverse_check = {}
        potential_duplicates = []

        for orig_col, canon_col in new_mapping.items():
            if canon_col in reverse_check:
                potential_duplicates.append(canon_col)
            else:
                reverse_check[canon_col] = orig_col

        if potential_duplicates:
            st.warning("⚠️ **Duplicate Mappings Detected**")
            st.write("The following canonical fields have multiple source columns:")
            for dup_field in set(potential_duplicates):
                source_cols = [k for k, v in new_mapping.items()
                               if v == dup_field]
                st.write(f"• **{dup_field}** ← {', '.join(source_cols)}")
            st.write(
                "You'll be able to resolve these conflicts after clicking 'Confirm & Apply Mapping'.")

    # Apply mapping button
    if st.button("✅ Confirm & Apply Mapping", type="primary"):
        if not new_mapping:
            st.warning(
                "⚠️ No mappings configured. Please select at least one field to map.")
        else:
            # Check for duplicate mappings
            reverse_mapping = {}
            duplicates = []

            for original_col, canonical_col in new_mapping.items():
                if canonical_col in reverse_mapping:
                    # Find existing duplicate entry or create new one
                    existing_dup = next(
                        (d for d in duplicates if d['canonical'] == canonical_col), None)
                    if existing_dup:
                        existing_dup['columns'].append(original_col)
                    else:
                        duplicates.append({
                            'canonical': canonical_col,
                            'columns': [reverse_mapping[canonical_col], original_col]
                        })
                else:
                    reverse_mapping[canonical_col] = original_col

            if duplicates:
                st.error("❌ Duplicate mappings detected!")
                st.write(
                    "**The following canonical fields are mapped to multiple columns:**")

                for dup in duplicates:
                    st.write(
                        f"• **{dup['canonical']}** ← {', '.join(dup['columns'])}")

                st.write("**Resolution Options:**")
                for i, dup in enumerate(duplicates):
                    st.write(f"\n**For field '{dup['canonical']}':**")

                    # Show data preview for each conflicting column
                    preview_cols = st.columns(len(dup['columns']))
                    for j, col in enumerate(dup['columns']):
                        with preview_cols[j]:
                            with st.expander(f"Preview '{col}'"):
                                st.write(df_to_use[col].head(5))

                    # Let user choose which column to keep
                    chosen_col = st.selectbox(
                        f"Which column should be used for '{dup['canonical']}'?",
                        options=dup['columns'] + ["Skip this field"],
                        key=f"resolve_{dup['canonical']}_{i}"
                    )

                    if chosen_col != "Skip this field":
                        # Remove other mappings for this canonical field
                        for col in dup['columns']:
                            if col != chosen_col and col in new_mapping:
                                del new_mapping[col]
                    else:
                        # Remove all mappings for this canonical field
                        for col in dup['columns']:
                            if col in new_mapping:
                                del new_mapping[col]

                st.write("---")
                st.info(
                    "Please resolve duplicates above, then click 'Confirm & Apply Mapping' again.")

            else:
                # No duplicates, proceed with mapping
                try:
                    with st.spinner("Applying mappings and final processing..."):
                        # Apply mapping to the processed DataFrame
                        mapped_df = df_to_use.rename(columns=new_mapping)

                        # Apply additional preprocessing to mapped data if field mapping is available
                        if new_mapping:
                            # Create reverse mapping for preprocessing
                            field_mapping = {
                                col: canonical for col, canonical in new_mapping.items()}

                            # Apply field-specific preprocessing
                            final_df, final_report = preprocess_dataframe(
                                mapped_df, field_mapping=field_mapping, config=preprocessing_config
                            )

                            st.session_state['final_df'] = final_df
                            st.session_state['final_report'] = final_report
                            st.session_state['column_mapping'] = new_mapping
                        else:
                            final_df = mapped_df
                            st.session_state['final_df'] = final_df
                            st.session_state['column_mapping'] = new_mapping

                        # Save user-approved mapping to history with pattern learning
                        for original_col, canonical_col in new_mapping.items():
                            col_key = original_col.lower().strip()
                            existing_history = history.get(col_key, {})
                            if isinstance(existing_history, dict):
                                approved_count = existing_history.get(
                                    'approved_count', 0) + 1
                            else:
                                approved_count = 1

                            history[col_key] = {
                                'canonical': canonical_col,
                                'original': original_col,
                                'approved_count': approved_count
                            }

                        save_history(history)

                        st.success(
                            "✅ Mapping applied and learned for future suggestions!")

                    # Show final processing report if available
                    if 'final_report' in st.session_state:
                        final_report = st.session_state['final_report']
                        if final_report.get('changes'):
                            with st.expander("🔧 Final Processing Report", expanded=False):
                                st.write(
                                    "**Additional field-specific processing:**")
                                for change in final_report['changes']:
                                    st.write(f"• {change}")

                    st.write("### 📊 Final Mapped Data Preview")
                    st.dataframe(final_df.head(), use_container_width=True)

                    st.write("### 📋 Mapping Summary")
                    mapping_summary = pd.DataFrame([
                        {
                            'Original Column': orig,
                            'Mapped To': mapped,
                            'Description': CANONICAL_SCHEMA[mapped],
                            'Data Type': str(final_df[mapped].dtype)
                        }
                        for orig, mapped in new_mapping.items()
                    ])
                    st.dataframe(mapping_summary, use_container_width=True)

                    # Data quality summary
                    st.write("### 📈 Data Quality Summary")
                    quality_metrics = []
                    for col in final_df.columns:
                        if col in CANONICAL_SCHEMA:
                            missing_pct = (
                                final_df[col].isnull().sum() / len(final_df)) * 100
                            unique_count = final_df[col].nunique()

                            quality_metrics.append({
                                'Field': col,
                                'Missing %': f"{missing_pct:.1f}%",
                                'Unique Values': unique_count,
                                'Data Type': str(final_df[col].dtype)
                            })

                    if quality_metrics:
                        quality_df = pd.DataFrame(quality_metrics)
                        st.dataframe(quality_df, use_container_width=True)

                    # Download options
                    st.write("### 📥 Download Options")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        #
                        # Download mapped CSV
                        csv = final_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Final CSV",
                            data=csv,
                            file_name="final_mapped_data.csv",
                            mime="text/csv"
                        )

                    with col2:
                        # Download mapping configuration
                        mapping_config = {
                            'column_mapping': new_mapping,
                            'preprocessing_config': preprocessing_config,
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'original_columns': list(df.columns),
                            'final_columns': list(final_df.columns)
                        }
                        config_json = json.dumps(mapping_config, indent=2)
                        st.download_button(
                            "⬇️ Download Config",
                            data=config_json,
                            file_name="mapping_config.json",
                            mime="application/json"
                        )

                    with col3:
                        # Download processing report
                        if st.session_state['preprocessing_report']:
                            report_data = {
                                'Stage': ['Original Data', 'After Preprocessing', 'After Mapping'],
                                'Rows': [
                                    st.session_state['preprocessing_report']['original_shape'][0],
                                    st.session_state['preprocessing_report']['final_shape'][0],
                                    len(final_df)
                                ],
                                'Columns': [
                                    st.session_state['preprocessing_report']['original_shape'][1],
                                    st.session_state['preprocessing_report']['final_shape'][1],
                                    len(final_df.columns)
                                ]
                            }
                            report_df = pd.DataFrame(report_data)
                            report_csv = report_df.to_csv(
                                index=False).encode("utf-8")
                            st.download_button(
                                "⬇️ Download Report",
                                data=report_csv,
                                file_name="processing_report.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"❌ Error applying mapping: {str(e)}")
                    st.write("Please check your column mappings and try again.")

                    # Show debug information
                    with st.expander("🐛 Debug Information"):
                        st.write("**New Mapping:**", new_mapping)
                        st.write("**DataFrame Columns:**",
                                 list(df_to_use.columns))
                        st.write("**Error Details:**", str(e))
                        import traceback
                        st.code(traceback.format_exc())

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload a CSV or XLSX file to get started")
