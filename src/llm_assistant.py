import os
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import BaseOutputParser
import json
import re
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MappingOutputParser(BaseOutputParser):
    """Custom parser for LLM mapping suggestions"""

    def parse(self, text: str) -> Dict:
        """Parse LLM output into structured mapping suggestions"""
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)

            # Fallback: parse line by line
            suggestions = {}
            lines = text.split('\n')

            for line in lines:
                if '→' in line or '->' in line:
                    # Parse format like "column_name → canonical_field"
                    parts = re.split(r'[→->]', line.strip())
                    if len(parts) == 2:
                        original = parts[0].strip().strip('"\'')
                        suggested = parts[1].strip().strip('"\'')
                        suggestions[original] = suggested

            return {'suggestions': suggestions, 'raw_response': text}

        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            return {'suggestions': {}, 'raw_response': text, 'error': str(e)}


class LLMColumnMappingAssistant:
    """LLM-powered assistant for column mapping validation and suggestions"""

    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            temperature=0.3,  # Lower temperature for more consistent mapping
            max_output_tokens=2048
        )

        self.output_parser = MappingOutputParser()

        # Create prompt template
        self.mapping_prompt = PromptTemplate.from_template("""
You are an expert data analyst specializing in e-commerce and order data mapping. Your task is to review and improve column mapping suggestions.

CANONICAL SCHEMA (Target Fields):
{canonical_schema}

ORIGINAL DATA COLUMNS:
{original_columns}

CURRENT MAPPING SUGGESTIONS:
{current_suggestions}

SAMPLE DATA (first 5 rows):
{sample_data}

TASK:
1. Review the current mapping suggestions
2. Analyze the sample data to understand column contents
3. Identify any incorrect mappings
4. Suggest improvements or corrections
5. Provide confidence scores (0-1) for each suggestion

OUTPUT FORMAT (JSON):
{{
  "improved_mappings": {{
    "column_name": {{
      "suggested_field": "canonical_field_name",
      "confidence": 0.95,
      "reason": "Brief explanation why this mapping is correct",
      "changes_from_original": "What changed and why"
    }}
  }},
  "validation_summary": {{
    "total_columns": 10,
    "mappings_approved": 8,
    "mappings_changed": 2,
    "unmapped_columns": 0,
    "overall_confidence": 0.87
  }},
  "recommendations": [
    "Any additional recommendations for data quality or mapping"
  ]
}}

Focus on:
- Data content matching field purpose
- Pattern recognition (IDs, dates, amounts, etc.)
- Missing or incorrect mappings
- Data quality issues
""")

        self.chain = self.mapping_prompt | self.llm | self.output_parser

    def validate_and_improve_mappings(self,
                                      original_columns: List[str],
                                      current_mappings: Dict[str, Dict],
                                      sample_data: pd.DataFrame,
                                      canonical_schema: Dict[str, str]) -> Dict:
        """
        Use LLM to validate and improve column mapping suggestions

        Args:
            original_columns: List of original column names
            current_mappings: Current mapping suggestions with confidence
            sample_data: Sample of the actual data
            canonical_schema: Target schema definition

        Returns:
            Dict with improved mappings and validation results
        """
        try:
            # Prepare sample data for LLM (limit to first 5 rows, key columns)
            sample_dict = {}
            # Limit columns to avoid token limits
            for col in original_columns[:10]:
                if col in sample_data.columns:
                    # Get first 5 non-null values
                    values = sample_data[col].dropna().head(5).tolist()
                    sample_dict[col] = values

            # Format current suggestions for LLM
            current_suggestions_formatted = {}
            for col, suggestion in current_mappings.items():
                if suggestion.get('field'):
                    current_suggestions_formatted[col] = {
                        'field': suggestion['field'],
                        'confidence': suggestion.get('confidence', 0),
                        'reason': suggestion.get('reason', '')
                    }

            # Invoke LLM
            response = self.chain.invoke({
                'canonical_schema': json.dumps(canonical_schema, indent=2),
                'original_columns': ', '.join(original_columns),
                'current_suggestions': json.dumps(current_suggestions_formatted, indent=2),
                'sample_data': json.dumps(sample_dict, indent=2),
            })

            logger.info("LLM validation completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            return {
                'suggestions': {},
                'error': str(e),
                'raw_response': ''
            }

    def get_field_suggestions_for_unmapped(self,
                                           unmapped_columns: List[str],
                                           sample_data: pd.DataFrame,
                                           canonical_schema: Dict[str, str]) -> Dict:
        """
        Get LLM suggestions for completely unmapped columns
        """
        if not unmapped_columns:
            return {'suggestions': {}}

        # Create a focused prompt for unmapped columns
        unmapped_prompt = PromptTemplate.from_template("""
You are a data mapping expert. Help map these unmapped columns to the canonical schema.

CANONICAL SCHEMA:
{canonical_schema}

UNMAPPED COLUMNS WITH SAMPLE DATA:
{unmapped_data}

For each unmapped column, suggest the best canonical field match.

OUTPUT FORMAT (JSON):
{{
  "suggestions": {{
    "column_name": {{
      "suggested_field": "canonical_field",
      "confidence": 0.85,
      "reason": "Why this mapping makes sense"
    }}
  }}
}}
""")

        try:
            # Prepare sample data for unmapped columns
            unmapped_data = {}
            for col in unmapped_columns:
                if col in sample_data.columns:
                    values = sample_data[col].dropna().head(3).tolist()
                    unmapped_data[col] = values

            chain = unmapped_prompt | self.llm | self.output_parser

            response = chain.invoke({
                'canonical_schema': json.dumps(canonical_schema, indent=2),
                'unmapped_data': json.dumps(unmapped_data, indent=2)
            })

            return response

        except Exception as e:
            logger.error(
                f"Error getting suggestions for unmapped columns: {e}")
            return {'suggestions': {}, 'error': str(e)}

    def explain_mapping_choice(self,
                               column_name: str,
                               suggested_field: str,
                               sample_values: List,
                               canonical_schema: Dict[str, str]) -> str:
        """
        Get detailed explanation for a specific mapping choice
        """
        explanation_prompt = PromptTemplate.from_template("""
Explain why the column "{column_name}" with sample values {sample_values} 
should be mapped to the canonical field "{suggested_field}".

Canonical field description: {field_description}

Provide a clear, concise explanation focusing on:
1. How the column name relates to the canonical field
2. How the sample data supports this mapping
3. Any data patterns that confirm the mapping

Keep the explanation under 100 words.
""")

        try:
            chain = explanation_prompt | self.llm | StrOutputParser()

            response = chain.invoke({
                'column_name': column_name,
                'sample_values': str(sample_values[:3]),  # Limit sample values
                'suggested_field': suggested_field,
                'field_description': canonical_schema.get(suggested_field, 'No description available')
            })

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Unable to generate explanation: {str(e)}"


def create_llm_assistant(google_api_key: str) -> Optional[LLMColumnMappingAssistant]:
    """
    Factory function to create LLM assistant with error handling
    """
    try:
        return LLMColumnMappingAssistant(google_api_key)
    except Exception as e:
        logger.error(f"Failed to create LLM assistant: {e}")
        return None
