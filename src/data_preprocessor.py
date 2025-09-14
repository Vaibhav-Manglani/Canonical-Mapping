import pandas as pd
import numpy as np
import re
from datetime import datetime, date
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for e-commerce/order data
    """

    def __init__(self):
        self.preprocessing_report = {
            'original_shape': None,
            'final_shape': None,
            'changes': [],
            'warnings': [],
            'data_quality_issues': []
        }

        # Define field-specific preprocessing rules
        self.field_processors = {
            'order_id': self._process_order_id,
            'order_date': self._process_date,
            'customer_id': self._process_customer_id,
            'customer_name': self._process_name,
            'email': self._process_email,
            'phone': self._process_phone,
            'billing_address': self._process_address,
            'shipping_address': self._process_address,
            'city': self._process_city,
            'state': self._process_state,
            'postal_code': self._process_postal_code,
            'country': self._process_country,
            'product_sku': self._process_sku,
            'product_name': self._process_product_name,
            'category': self._process_category,
            'subcategory': self._process_category,
            'quantity': self._process_numeric,
            'unit_price': self._process_currency,
            'currency': self._process_currency_code,
            'discount_pct': self._process_percentage,
            'tax_pct': self._process_percentage,
            'shipping_fee': self._process_currency,
            'total_amount': self._process_currency,
            'tax_id': self._process_tax_id
        }

    def preprocess_data(self, df: pd.DataFrame,
                        field_mapping: Dict[str, str] = None,
                        config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Main preprocessing function

        Args:
            df: Input DataFrame
            field_mapping: Mapping of column names to canonical fields
            config: Configuration parameters

        Returns:
            Tuple of (processed_df, preprocessing_report)
        """
        if config is None:
            config = self._get_default_config()

        self.preprocessing_report['original_shape'] = df.shape
        processed_df = df.copy()

        logger.info(f"Starting preprocessing of data with shape: {df.shape}")

        # 1. Handle completely empty columns
        processed_df = self._remove_empty_columns(processed_df)

        # 2. Handle duplicate rows
        processed_df = self._handle_duplicates(processed_df, config)

        # 3. Apply field-specific preprocessing if mapping is provided
        if field_mapping:
            processed_df = self._apply_field_specific_processing(
                processed_df, field_mapping, config)
        else:
            # Generic preprocessing
            processed_df = self._apply_generic_preprocessing(
                processed_df, config)

        # 4. Handle missing values
        processed_df = self._handle_missing_values(processed_df, config)

        # 5. Validate data quality
        self._validate_data_quality(processed_df, field_mapping)

        # 6. Generate final report
        self.preprocessing_report['final_shape'] = processed_df.shape
        logger.info(
            f"Preprocessing completed. Final shape: {processed_df.shape}")

        return processed_df, self.preprocessing_report

    def _get_default_config(self) -> Dict[str, Any]:
        """Default preprocessing configuration"""
        return {
            'remove_duplicates': True,
            'handle_missing': True,
            'standardize_formats': True,
            'validate_data': True,
            'missing_threshold': 0.8,  # Remove columns with >80% missing
            'outlier_detection': True,
            'date_format': 'infer',
            'currency_symbol': '₹',
            'phone_country_code': '+91',
            'default_country': 'India'
        }

    def _remove_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty columns"""
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            self.preprocessing_report['changes'].append(
                f"Removed {len(empty_cols)} empty columns: {empty_cols}")
        return df

    def _handle_duplicates(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Handle duplicate rows"""
        if config.get('remove_duplicates', True):
            initial_count = len(df)
            df = df.drop_duplicates()
            duplicate_count = initial_count - len(df)
            if duplicate_count > 0:
                self.preprocessing_report['changes'].append(
                    f"Removed {duplicate_count} duplicate rows")
        return df

    def _apply_field_specific_processing(self, df: pd.DataFrame,
                                         field_mapping: Dict[str, str],
                                         config: Dict) -> pd.DataFrame:
        """Apply field-specific preprocessing based on canonical field types"""
        for column, canonical_field in field_mapping.items():
            if column in df.columns and canonical_field in self.field_processors:
                try:
                    df[column] = self.field_processors[canonical_field](
                        df[column], config)
                    logger.info(f"Processed {column} as {canonical_field}")
                except Exception as e:
                    warning_msg = f"Error processing {column} as {canonical_field}: {str(e)}"
                    self.preprocessing_report['warnings'].append(warning_msg)
                    logger.warning(warning_msg)
        return df

    def _apply_generic_preprocessing(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Apply generic preprocessing when no field mapping is available"""
        for column in df.columns:
            # Auto-detect and process common patterns
            if self._looks_like_date(df[column]):
                df[column] = self._process_date(df[column], config)
            elif self._looks_like_phone(df[column]):
                df[column] = self._process_phone(df[column], config)
            elif self._looks_like_email(df[column]):
                df[column] = self._process_email(df[column], config)
            elif self._looks_like_currency(df[column]):
                df[column] = self._process_currency(df[column], config)
        return df

    def _handle_missing_values(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        if not config.get('handle_missing', True):
            return df

        missing_threshold = config.get('missing_threshold', 0.8)

        # Remove columns with too many missing values
        missing_ratios = df.isnull().sum() / len(df)
        cols_to_remove = missing_ratios[missing_ratios >
                                        missing_threshold].index.tolist()

        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            self.preprocessing_report['changes'].append(
                f"Removed columns with >{missing_threshold*100}% missing values: {cols_to_remove}"
            )

        # Fill remaining missing values intelligently
        for column in df.columns:
            if df[column].isnull().any():
                df[column] = self._fill_missing_values(df[column], column)

        return df

    def _fill_missing_values(self, series: pd.Series, column_name: str) -> pd.Series:
        """Intelligent missing value imputation"""
        if series.dtype == 'object':
            # For text columns, fill with 'Unknown' or most frequent value
            mode_val = series.mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            return series.fillna(fill_val)
        elif pd.api.types.is_numeric_dtype(series):
            # For numeric columns, fill with median
            return series.fillna(series.median())
        else:
            return series.fillna('Unknown')

    # Field-specific processors
    def _process_order_id(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process order ID fields"""
        return series.astype(str).str.strip().str.upper()

    def _process_date(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process date fields with multiple format support"""
        def parse_date(date_str):
            if pd.isna(date_str) or date_str == '':
                return pd.NaT

            date_str = str(date_str).strip()

            # Common date formats
            formats = [
                '%d-%b-%y', '%d-%b-%Y',  # 02-Aug-25, 02-Aug-2025
                '%d/%m/%Y', '%d/%m/%y',   # 02/08/2025, 02/08/25
                '%Y-%m-%d',               # 2025-08-02
                '%m/%d/%Y', '%m/%d/%y',   # 08/02/2025, 08/02/25
                '%d.%m.%Y', '%d.%m.%y'    # 02.08.2025, 02.08.25
            ]

            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue

            # Try pandas auto-parsing as last resort
            try:
                return pd.to_datetime(date_str)
            except:
                return pd.NaT

        return series.apply(parse_date)

    def _process_customer_id(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process customer ID fields"""
        return series.astype(str).str.strip().str.upper()

    def _process_name(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process name fields"""
        def clean_name(name):
            if pd.isna(name) or name == '':
                return name
            name = str(name).strip()
            # Capitalize each word
            return ' '.join(word.capitalize() for word in name.split())

        return series.apply(clean_name)

    def _process_email(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process email fields"""
        def validate_email(email):
            if pd.isna(email) or email == '':
                return email
            email = str(email).strip().lower()
            # Basic email validation
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(email_pattern, email):
                return email
            else:
                return 'invalid_email'

        return series.apply(validate_email)

    def _process_phone(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process phone number fields"""
        def clean_phone(phone):
            if pd.isna(phone) or phone == '':
                return phone

            phone = str(phone).strip()
            # Remove all non-digit characters except + at the beginning
            phone = re.sub(r'[^\d+]', '', phone)

            # Add country code if missing
            if phone and not phone.startswith('+'):
                country_code = config.get('phone_country_code', '+91')
                if len(phone) == 10:  # Indian mobile number
                    phone = country_code + phone

            return phone

        return series.apply(clean_phone)

    def _process_address(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process address fields"""
        def clean_address(addr):
            if pd.isna(addr) or addr == '':
                return addr
            addr = str(addr).strip()
            # Remove extra spaces
            return re.sub(r'\s+', ' ', addr)

        return series.apply(clean_address)

    def _process_city(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process city fields"""
        return self._process_name(series, config)  # Same as name processing

    def _process_state(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process state fields"""
        return self._process_name(series, config)

    def _process_postal_code(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process postal code fields"""
        def clean_postal(postal):
            if pd.isna(postal) or postal == '':
                return postal
            postal = str(postal).strip()
            # Handle Indian postal codes with XX pattern
            if 'XX' in postal:
                return postal.upper()
            # Ensure numeric postal codes are strings
            return postal

        return series.apply(clean_postal)

    def _process_country(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process country fields"""
        def standardize_country(country):
            if pd.isna(country) or country == '':
                return config.get('default_country', 'India')
            return str(country).strip().title()

        return series.apply(standardize_country)

    def _process_sku(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process SKU fields"""
        return series.astype(str).str.strip().str.upper()

    def _process_product_name(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process product name fields"""
        return self._process_name(series, config)

    def _process_category(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process category fields"""
        return self._process_name(series, config)

    def _process_numeric(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process numeric fields"""
        def clean_numeric(value):
            if pd.isna(value):
                return np.nan
            try:
                # Remove any non-numeric characters except decimal point
                cleaned = re.sub(r'[^\d.-]', '', str(value))
                return float(cleaned) if cleaned else np.nan
            except:
                return np.nan

        result = series.apply(clean_numeric)
        return pd.to_numeric(result, errors='coerce')

    def _process_currency(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process currency fields"""
        return self._process_numeric(series, config)

    def _process_currency_code(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process currency code fields"""
        def standardize_currency(curr):
            if pd.isna(curr) or curr == '':
                return 'INR'  # Default currency
            return str(curr).strip().upper()

        return series.apply(standardize_currency)

    def _process_percentage(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process percentage fields"""
        def clean_percentage(value):
            if pd.isna(value):
                return np.nan
            try:
                value = str(value).strip()
                # Remove % symbol if present
                value = value.replace('%', '')
                cleaned = float(value)
                # Convert to decimal if it looks like a percentage (>1)
                if cleaned > 1:
                    cleaned = cleaned / 100
                return cleaned
            except:
                return np.nan

        return series.apply(clean_percentage)

    def _process_tax_id(self, series: pd.Series, config: Dict) -> pd.Series:
        """Process tax ID fields (GSTIN, etc.)"""
        def clean_tax_id(tax_id):
            if pd.isna(tax_id) or tax_id == '':
                return tax_id
            return str(tax_id).strip().upper()

        return series.apply(clean_tax_id)

    # Helper methods for auto-detection
    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if series looks like dates"""
        sample = series.dropna().astype(str).head(10)
        date_patterns = [
            r'\d{1,2}[-/]\w{3}[-/]\d{2,4}',  # 02-Aug-25
            r'\d{4}-\d{2}-\d{2}',             # 2025-08-02
            r'\d{1,2}/\d{1,2}/\d{2,4}'       # 02/08/25
        ]

        matches = 0
        for pattern in date_patterns:
            matches += sample.str.contains(pattern, regex=True, na=False).sum()

        return matches > len(sample) * 0.5

    def _looks_like_phone(self, series: pd.Series) -> bool:
        """Check if series looks like phone numbers"""
        sample = series.dropna().astype(str).head(10)
        phone_pattern = r'^[-+]?\d{10,}$'
        matches = sample.str.contains(
            phone_pattern, regex=True, na=False).sum()
        return matches > len(sample) * 0.5

    def _looks_like_email(self, series: pd.Series) -> bool:
        """Check if series looks like emails"""
        sample = series.dropna().astype(str).head(10)
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        matches = sample.str.contains(
            email_pattern, regex=True, na=False).sum()
        return matches > len(sample) * 0.5

    def _looks_like_currency(self, series: pd.Series) -> bool:
        """Check if series looks like currency values"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            non_null_ratio = numeric_series.notna().sum() / len(series)
            return non_null_ratio > 0.7 and numeric_series.min() >= 0
        except:
            return False

    def _validate_data_quality(self, df: pd.DataFrame, field_mapping: Dict = None):
        """Validate data quality and add issues to report"""
        issues = []

        # Check for high missing value ratios
        missing_ratios = df.isnull().sum() / len(df)
        high_missing = missing_ratios[missing_ratios > 0.5]
        if not high_missing.empty:
            issues.append(f"High missing values in: {high_missing.to_dict()}")

        # Check for potential data quality issues
        for column in df.columns:
            # Check for suspicious patterns
            if df[column].dtype == 'object':
                value_counts = df[column].value_counts()
                if len(value_counts) == 1 and not df[column].isnull().all():
                    issues.append(
                        f"Column '{column}' has only one unique value")

        self.preprocessing_report['data_quality_issues'] = issues

    def get_preprocessing_summary(self) -> Dict:
        """Get a summary of preprocessing operations"""
        return {
            'shape_change': f"{self.preprocessing_report['original_shape']} → {self.preprocessing_report['final_shape']}",
            'changes_made': len(self.preprocessing_report['changes']),
            'warnings': len(self.preprocessing_report['warnings']),
            'data_quality_issues': len(self.preprocessing_report['data_quality_issues']),
            'details': self.preprocessing_report
        }


# Convenience function for easy usage
def preprocess_dataframe(df: pd.DataFrame,
                         field_mapping: Dict[str, str] = None,
                         config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to preprocess a DataFrame

    Usage:
        processed_df, report = preprocess_dataframe(df, field_mapping)
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_data(df, field_mapping, config)
