import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime

HISTORY_FILE = "history.json"

SEMANTIC_PATTERNS = {
    "order_id": {
        "keywords": ["order", "ord", "reference", "ref", "id", "number", "no"],
        "patterns": [r"ord[-_]?\d+", r"ref[-_]?\d+", r"order[-_]?id"],
        "data_patterns": [r"^[A-Z]+-\d+$", r"^\d+$"]
    },
    "order_date": {
        "keywords": ["date", "ordered", "created", "placed", "time"],
        "patterns": [r".*date.*", r".*ordered.*", r".*created.*"],
        "data_patterns": [r"\d{1,2}[-/]\w{3}[-/]\d{2,4}", r"\d{4}-\d{2}-\d{2}"]
    },
    "customer_id": {
        "keywords": ["customer", "client", "cust", "user", "account"],
        "patterns": [r"(customer|client|cust)[-_]?(id|ref|no)", r"account[-_]?id"],
        "data_patterns": [r"^CUST-\d+$", r"^CLIENT-\d+$", r"^ACC\d+$"]
    },
    "customer_name": {
        "keywords": ["name", "customer", "client", "full", "contact"],
        "patterns": [r"(customer|client)[-_]?name", r"full[-_]?name", r"contact[-_]?name"],
        # First Last name pattern
        "data_patterns": [r"^[A-Z][a-z]+ [A-Z][a-z]+"]
    },
    "email": {
        "keywords": ["email", "mail", "contact", "e-mail"],
        "patterns": [r".*email.*", r".*mail.*", r"contact.*"],
        "data_patterns": [r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"]
    },
    "phone": {
        "keywords": ["phone", "mobile", "tel", "contact", "number"],
        "patterns": [r".*phone.*", r".*mobile.*", r".*tel.*"],
        "data_patterns": [r"[-+]?\d{10,}", r"\(\d{3}\)\s*\d{3}-\d{4}"]
    },
    "billing_address": {
        "keywords": ["bill", "billing", "invoice", "payment"],
        "patterns": [r"bill[-_]?(to|address)", r"billing[-_]?address", r"invoice[-_]?address"],
        "data_patterns": [r"\d+\s+[A-Za-z\s]+"]  # Address with number
    },
    "shipping_address": {
        "keywords": ["ship", "shipping", "delivery", "send"],
        "patterns": [r"ship[-_]?(to|address)", r"shipping[-_]?address", r"delivery[-_]?address"],
        "data_patterns": [r"\d+\s+[A-Za-z\s]+"]
    },
    "city": {
        "keywords": ["city", "town", "location"],
        "patterns": [r".*city.*", r".*town.*", r".*location.*"],
        "data_patterns": [r"^[A-Z][a-z]+$"]  # Capitalized city names
    },
    "postal_code": {
        "keywords": ["postal", "zip", "pin", "code"],
        "patterns": [r"postal[-_]?code", r"zip[-_]?code", r"pin[-_]?code"],
        "data_patterns": [r"^\d{5,6}[A-Z]{0,2}\d?$", r"^\d{6}XX\d$"]
    },
    "product_sku": {
        "keywords": ["sku", "stock", "product", "item", "code"],
        "patterns": [r"stock[-_]?code", r"product[-_]?code", r"item[-_]?code", r"sku"],
        "data_patterns": [r"^[A-Z]{2}-\d{4}$", r"^[A-Z]+\d+$"]
    },
    "product_name": {
        "keywords": ["product", "item", "desc", "description", "name", "title"],
        "patterns": [r"product[-_]?name", r"item[-_]?name", r"description", r"desc"],
        "data_patterns": [r"^[A-Za-z\s]+$"]  # Alphabetic product names
    },
    "quantity": {
        "keywords": ["qty", "quantity", "amount", "count", "units"],
        "patterns": [r"qty", r"quantity", r"amount", r"count"],
        "data_patterns": [r"^\d+$"]  # Simple integers
    },
    "unit_price": {
        "keywords": ["price", "cost", "rate", "amount", "value"],
        "patterns": [r"unit[-_]?price", r"price[-_]?per", r"rate", r"cost"],
        "data_patterns": [r"^\d+\.?\d*$"]  # Price numbers
    },
    "discount_pct": {
        "keywords": ["discount", "off", "reduction", "deduction"],
        "patterns": [r"discount", r".*off.*", r"reduction"],
        "data_patterns": [r"^0\.\d+$", r"^\d{1,2}%?$"]  # Percentage format
    },
    "tax_pct": {
        "keywords": ["tax", "gst", "vat", "rate"],
        "patterns": [r"tax", r"gst", r"vat"],
        "data_patterns": [r"^0\.\d+$", r"^\d{1,2}%?$"]
    },
    "shipping_fee": {
        "keywords": ["shipping", "logistics", "delivery", "freight", "postage"],
        "patterns": [r"shipping[-_]?fee", r"logistics[-_]?fee", r"delivery[-_]?fee"],
        "data_patterns": [r"^\d+\.?\d*$"]
    },
    "total_amount": {
        "keywords": ["total", "grand", "final", "amount", "sum"],
        "patterns": [r"grand[-_]?total", r"total[-_]?amount", r"final[-_]?amount"],
        "data_patterns": [r"^\d+\.?\d*$"]
    },
    "tax_id": {
        "keywords": ["gstin", "tax", "vat", "id", "number"],
        "patterns": [r"gstin", r"tax[-_]?id", r"vat[-_]?id"],
        "data_patterns": [r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z]\d$"]
    }
}


def analyze_data_content(column_name, sample_data, max_samples=100):
    """Analyze actual data content to determine field type"""
    if sample_data.empty:
        return {}

    # Get non-null samples
    samples = sample_data.dropna().astype(str).head(max_samples)
    if len(samples) == 0:
        return {}

    analysis = {
        'sample_count': len(samples),
        'null_ratio': sample_data.isnull().sum() / len(sample_data),
        'unique_ratio': sample_data.nunique() / len(sample_data),
        'patterns_found': []
    }

    # Check for specific data patterns
    for canonical_field, patterns in SEMANTIC_PATTERNS.items():
        pattern_matches = 0
        for data_pattern in patterns.get('data_patterns', []):
            pattern_matches += samples.str.contains(
                data_pattern, regex=True, na=False).sum()

        if pattern_matches > 0:
            analysis['patterns_found'].append({
                'field': canonical_field,
                'matches': pattern_matches,
                'match_ratio': pattern_matches / len(samples)
            })

    # Sort by match ratio
    analysis['patterns_found'].sort(
        key=lambda x: x['match_ratio'], reverse=True)

    return analysis


def calculate_semantic_similarity(column_name, canonical_field):
    """Calculate semantic similarity between column name and canonical field"""
    col_lower = column_name.lower().strip()
    patterns = SEMANTIC_PATTERNS.get(canonical_field, {})

    score = 0.0
    reasons = []

    # Check keyword matches
    keywords = patterns.get('keywords', [])
    for keyword in keywords:
        if keyword in col_lower:
            score += 0.3
            reasons.append(f"Contains keyword '{keyword}'")

    # Check pattern matches
    regex_patterns = patterns.get('patterns', [])
    for pattern in regex_patterns:
        if re.search(pattern, col_lower):
            score += 0.4
            reasons.append(f"Matches pattern '{pattern}'")

    return min(score, 1.0), reasons


def suggest_mapping(input_columns, dataframe, history):
    """Enhanced mapping with data content analysis"""
    suggestions = {}

    for col in input_columns:
        col_lower = col.lower().strip()
        best_match = None
        best_score = 0.0
        best_reasons = []

        # 1. Check history first (with higher weight for frequently approved mappings)
        if col_lower in history:
            hist_entry = history[col_lower]
            if isinstance(hist_entry, dict):
                canonical = hist_entry.get('canonical')
                approved_count = hist_entry.get('approved_count', 1)
                # Higher confidence for frequently used
                confidence = min(0.9 + (approved_count * 0.02), 1.0)
                suggestions[col] = {
                    'field': canonical,
                    'confidence': confidence,
                    'reason': f"Historical mapping (used {approved_count} times)"
                }
                continue
            else:
                # Legacy format
                suggestions[col] = {
                    'field': hist_entry,
                    'confidence': 0.8,
                    'reason': "Historical mapping"
                }
                continue

        # 2. Analyze actual data content
        data_analysis = analyze_data_content(col, dataframe[col])

        # 3. Score all canonical fields
        for canonical_field in SEMANTIC_PATTERNS.keys():
            total_score = 0.0
            reasons = []

            # Semantic similarity score
            semantic_score, semantic_reasons = calculate_semantic_similarity(
                col, canonical_field)
            total_score += semantic_score * 0.6  # 60% weight for semantic matching
            reasons.extend(semantic_reasons)

            # Data pattern matching score
            data_score = 0.0
            if data_analysis.get('patterns_found'):
                for pattern_match in data_analysis['patterns_found']:
                    if pattern_match['field'] == canonical_field:
                        # 40% weight for data patterns
                        data_score = pattern_match['match_ratio'] * 0.4
                        reasons.append(
                            f"Data matches {canonical_field} patterns ({pattern_match['match_ratio']:.1%} of samples)")
                        break

            total_score += data_score

            # Bonus for high uniqueness ratio (good for IDs)
            if canonical_field in ['order_id', 'customer_id'] and data_analysis.get('unique_ratio', 0) > 0.8:
                total_score += 0.1
                reasons.append("High uniqueness suggests identifier field")

            # Penalty for high null ratio
            if data_analysis.get('null_ratio', 0) > 0.5:
                total_score *= 0.7
                reasons.append("High null ratio reduces confidence")

            if total_score > best_score and total_score > 0.3:  # Minimum threshold
                best_match = canonical_field
                best_score = total_score
                best_reasons = reasons

        if best_match:
            suggestions[col] = {
                'field': best_match,
                'confidence': min(best_score, 1.0),
                'reason': "; ".join(best_reasons[:2])  # Top 2 reasons
            }
        else:
            suggestions[col] = {
                'field': None,
                'confidence': 0.0,
                'reason': "No confident match found"
            }

    return suggestions


def load_history():
    """Load mapping history from file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_history(history):
    """Save mapping history to file"""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
