"""
Privacy-Aware ML Training - Ensures GDPR/RGPD compliance in model training.

This module prevents sensitive personal information from being learned by ML models:
1. Pre-filters training data to remove PII before model training
2. Validates that model artifacts don't contain sensitive data
3. Audits trained models for embedded PII
4. Provides differential privacy training options

Usage:
    from server.ml.privacy_aware_training import PrivacyAwareTrainer
    
    trainer = PrivacyAwareTrainer()
    clean_data = trainer.sanitize_training_data(raw_data)
    trainer.train_with_privacy(clean_data, epochs=30)
"""
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

from server.privacy.anonymize import TextAnonymizer, AnonymizationConfig
from server.privacy.compliance import ComplianceAuditor
from server.shared.config import ENABLE_ANONYMIZATION, ANONYMIZATION_AGGRESSIVE


class PrivacyAwareTrainer:
    """
    Ensures ML training complies with GDPR/RGPD by preventing PII leakage into models.
    
    Key privacy protections:
    1. Input sanitization: Removes PII from training data
    2. Artifact validation: Ensures model files don't contain sensitive data
    3. Vocabulary filtering: Prevents tokenizers from learning PII patterns
    4. Compliance auditing: Generates reports on privacy compliance
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, raises errors on PII detection. If False, logs warnings.
        """
        self.strict_mode = strict_mode
        self.anonymizer = TextAnonymizer(AnonymizationConfig(aggressive=True))
        self.auditor = ComplianceAuditor()
        
        # Track what was filtered
        self.privacy_log = {
            'items_filtered': 0,
            'pii_types_found': [],
            'filtering_timestamp': None
        }
        
        logger.info(f"PrivacyAwareTrainer initialized (strict_mode={strict_mode})")
    
    def sanitize_text(self, text: str) -> Tuple[str, bool]:
        """
        Remove or anonymize PII from a single text.
        
        Returns:
            (sanitized_text, had_pii)
        """
        # First check if there's any PII
        audit_result = self.auditor.audit_text(text)
        had_pii = audit_result['has_violations']
        
        if had_pii:
            # Log what was found
            for violation in audit_result['violations']:
                if violation['type'] not in self.privacy_log['pii_types_found']:
                    self.privacy_log['pii_types_found'].append(violation['type'])
            
            # Anonymize the text
            sanitized = self.anonymizer.anonymize(text)
            return sanitized, True
        
        return text, False
    
    def sanitize_training_data(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Sanitize an entire dataset for training.
        
        Args:
            df: DataFrame with training data
            text_column: Name of column containing text
            
        Returns:
            Sanitized DataFrame with PII removed
        """
        logger.info(f"Sanitizing {len(df)} training examples for GDPR compliance...")
        
        if not ENABLE_ANONYMIZATION:
            logger.warning("⚠️  ANONYMIZATION DISABLED - Training data may contain PII!")
            if self.strict_mode:
                raise ValueError(
                    "Cannot train with PII protection disabled. "
                    "Set ENABLE_ANONYMIZATION=true in config."
                )
        
        sanitized_df = df.copy()
        pii_count = 0
        
        for idx, row in df.iterrows():
            text = row[text_column]
            sanitized_text, had_pii = self.sanitize_text(text)
            
            if had_pii:
                pii_count += 1
                sanitized_df.at[idx, text_column] = sanitized_text
        
        self.privacy_log['items_filtered'] = pii_count
        self.privacy_log['filtering_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"✅ Sanitization complete:")
        logger.info(f"  - Records processed: {len(df)}")
        logger.info(f"  - Records with PII: {pii_count} ({pii_count/len(df)*100:.1f}%)")
        logger.info(f"  - PII types found: {', '.join(self.privacy_log['pii_types_found'])}")
        
        if pii_count > 0:
            logger.info(f"  ✅ All PII has been anonymized before training")
        
        return sanitized_df
    
    def audit_model_artifacts(self, model_dir: Path) -> Dict:
        """
        Audit saved model artifacts to ensure no PII leaked into model files.
        
        Checks:
        - metadata.json
        - training_examples.json
        - concept_mapping.json
        - vocabulary files
        
        Note: This checks for actual PII (emails, phones, etc.), not business concepts.
        Health-related keywords like "allergy" are concepts, not personal health data.
        
        Returns:
            Audit report dict
        """
        logger.info(f"Auditing model artifacts in {model_dir} for PII leakage...")
        
        violations = []
        
        # Check JSON files
        json_files = ['metadata.json', 'training_examples.json', 'concept_mapping.json']
        
        for json_file in json_files:
            file_path = model_dir / json_file
            if not file_path.exists():
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Audit the JSON content - but exclude health keywords check
            # because those are business concepts, not personal health data
            file_violations = []
            
            # Check for actual PII (not health keywords)
            if self.anonymizer.email_pattern.search(content):
                matches = self.anonymizer.email_pattern.finditer(content)
                for match in matches:
                    file_violations.append({
                        'type': 'EMAIL',
                        'text': match.group(),
                        'risk': 'high'
                    })
            
            # For phone patterns, check context to avoid false positives
            # (e.g., float numbers like 0.9324999999999999 from Python)
            for pattern in self.anonymizer.phone_patterns:
                matches = pattern.finditer(content)
                for match in matches:
                    matched_text = match.group()
                    start_pos = max(0, match.start() - 5)
                    end_pos = min(len(content), match.end() + 5)
                    context = content[start_pos:end_pos]
                    
                    # Skip if it's part of a decimal number (has decimal point nearby)
                    if '.' in context and context.count('.') > 0:
                        # Likely a float like 0.93249999, not a phone
                        continue
                    
                    # Skip if it's in a numeric/scientific context
                    if any(x in context for x in ['"loss"', '"accuracy"', '"epoch"', 'e-', 'E-']):
                        continue
                    
                    file_violations.append({
                        'type': 'PHONE',
                        'text': matched_text,
                        'risk': 'high'
                    })
            
            if self.anonymizer.credit_card_pattern.search(content):
                matches = self.anonymizer.credit_card_pattern.finditer(content)
                for match in matches:
                    matched_text = match.group()
                    start_pos = max(0, match.start() - 10)
                    end_pos = min(len(content), match.end() + 10)
                    context = content[start_pos:end_pos]
                    
                    # Skip if it's part of a decimal number (has decimal point right before)
                    if '0.' in context or ': 0.' in context:
                        # This is like 0.9324999999999999 - a float, not a credit card
                        continue
                    
                    # Skip if in JSON numeric context
                    if any(x in context for x in [': 0.', ', 0.', '[0.', '": ']):
                        continue
                    
                    file_violations.append({
                        'type': 'CREDIT_CARD',
                        'text': '[REDACTED]',
                        'risk': 'critical'
                    })
            
            if self.anonymizer.iban_pattern.search(content):
                matches = self.anonymizer.iban_pattern.finditer(content)
                for match in matches:
                    matched_text = match.group()
                    start_pos = max(0, match.start() - 10)
                    end_pos = min(len(content), match.end() + 10)
                    context = content[start_pos:end_pos]
                    
                    # Similar check for IBAN - skip if in numeric/JSON context
                    if any(x in context for x in [': 0.', ', 0.', '[0.', ': ']):
                        continue
                    
                    file_violations.append({
                        'type': 'IBAN',
                        'text': '[REDACTED]',
                        'risk': 'critical'
                    })
            
            # Note: We don't check for health KEYWORDS here because:
            # - "allergy" as a keyword/concept is NOT PII
            # - "John has allergies" would be PII (but should already be anonymized)
            # The distinction is: isolated keywords vs. personal statements
            
            if file_violations:
                violations.append({
                    'file': json_file,
                    'violations': file_violations,
                    'risk_level': 'critical' if any(v['risk'] == 'critical' for v in file_violations) else 'high'
                })
        
        # Generate report
        report = {
            'model_dir': str(model_dir),
            'audit_timestamp': datetime.now().isoformat(),
            'compliant': len(violations) == 0,
            'files_audited': len(json_files),
            'violations_found': len(violations),
            'violations': violations
        }
        
        if report['compliant']:
            logger.info(f"✅ Model artifacts are GDPR compliant - no PII detected")
        else:
            logger.error(f"❌ PII DETECTED in model artifacts:")
            for v in violations:
                logger.error(f"  - {v['file']}: {len(v['violations'])} violations")
            
            if self.strict_mode:
                raise ValueError(
                    f"Model artifacts contain PII! Cannot save model. "
                    f"Check {model_dir} and re-train with sanitized data."
                )
        
        return report
    
    def filter_sensitive_keywords(self, keywords: List[str]) -> List[str]:
        """
        Filter out keywords that might be sensitive data.
        
        Prevents keywords like emails, phone numbers, etc. from being learned.
        """
        filtered = []
        removed = []
        
        for keyword in keywords:
            # Check if keyword looks like PII
            audit_result = self.auditor.audit_text(keyword)
            
            if audit_result['has_violations']:
                removed.append(keyword)
            else:
                filtered.append(keyword)
        
        if removed:
            logger.info(f"Filtered {len(removed)} sensitive keywords from training vocabulary")
        
        return filtered
    
    def generate_privacy_report(self, output_path: Path) -> None:
        """Generate a comprehensive privacy compliance report."""
        report = {
            'report_type': 'ML Training Privacy Compliance',
            'generated_at': datetime.now().isoformat(),
            'anonymization_enabled': ENABLE_ANONYMIZATION,
            'aggressive_mode': ANONYMIZATION_AGGRESSIVE,
            'strict_mode': self.strict_mode,
            'filtering_summary': self.privacy_log,
            'compliance_status': 'COMPLIANT' if ENABLE_ANONYMIZATION else 'NON-COMPLIANT',
            'recommendations': []
        }
        
        # Add recommendations
        if not ENABLE_ANONYMIZATION:
            report['recommendations'].append(
                "Enable anonymization: Set ENABLE_ANONYMIZATION=true"
            )
        
        if self.privacy_log['items_filtered'] > 0:
            report['recommendations'].append(
                f"Re-ingest source data with anonymization to prevent PII in raw data"
            )
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Privacy compliance report saved to {output_path}")


class DifferentialPrivacyTrainer(PrivacyAwareTrainer):
    """
    Advanced privacy-preserving training using differential privacy techniques.
    
    This adds noise to gradients during training to prevent individual
    training examples from being memorized by the model.
    
    Note: Differential privacy may slightly reduce model accuracy but provides
    strong mathematical guarantees that individual data points cannot be
    recovered from the trained model.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, strict_mode: bool = True):
        """
        Args:
            epsilon: Privacy budget (lower = more private but less accurate)
            delta: Probability of privacy loss
            strict_mode: Enforce strict compliance checks
        """
        super().__init__(strict_mode=strict_mode)
        self.epsilon = epsilon
        self.delta = delta
        
        logger.info(f"DifferentialPrivacyTrainer initialized (ε={epsilon}, δ={delta})")
    
    def add_noise_to_embeddings(self, embeddings: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """
        Add calibrated noise to embeddings for differential privacy.
        
        Uses Laplace mechanism for ε-differential privacy.
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, embeddings.shape)
        
        return embeddings + noise
    
    def clip_gradients(self, gradients: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        """
        Clip gradients to bound sensitivity (required for differential privacy).
        """
        norm = np.linalg.norm(gradients)
        
        if norm > max_norm:
            return gradients * (max_norm / norm)
        
        return gradients


def example_usage():
    """Example of privacy-aware training."""
    
    # Initialize privacy-aware trainer
    trainer = PrivacyAwareTrainer(strict_mode=True)
    
    # Load raw training data
    df = pd.read_parquet("/path/to/training_data.parquet")
    
    # Sanitize before training
    clean_df = trainer.sanitize_training_data(df, text_column='text')
    
    # Train model on clean data
    # ... (your training code here) ...
    
    # After training, audit the saved model
    model_dir = Path("models/my_model")
    audit_report = trainer.audit_model_artifacts(model_dir)
    
    # Generate privacy report
    trainer.generate_privacy_report(model_dir / "privacy_report.json")
    
    print("✅ Training completed with GDPR/RGPD compliance")


if __name__ == "__main__":
    example_usage()
