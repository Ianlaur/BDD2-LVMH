"""
GDPR/RGPD Compliance CLI - Manage sensitive data protection and privacy compliance.

This tool helps ensure LVMH pipeline complies with GDPR/RGPD regulations by:
- Detecting and anonymizing sensitive personal information (PII)
- Auditing data for compliance violations
- Generating compliance reports
- Managing data retention policies
- Testing anonymization effectiveness

Usage:
    # Audit current data for sensitive information
    python -m server.privacy.compliance audit
    
    # Test anonymization on sample text
    python -m server.privacy.compliance test "Jean Dupont, email: jean@example.com, tel: 06 12 34 56 78"
    
    # Generate compliance report
    python -m server.privacy.compliance report --output compliance_report.pdf
    
    # Anonymize a specific file
    python -m server.privacy.compliance anonymize --file data/input/notes.csv --column Transcription
    
    # Check compliance configuration
    python -m server.privacy.compliance config
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

from server.privacy.anonymize import TextAnonymizer, AnonymizationConfig, _COMPILED_ARTICLE9
from server.shared.config import (
    DATA_PROCESSED, DATA_OUTPUTS, DATA_INPUT, BASE_DIR,
    ENABLE_ANONYMIZATION, ANONYMIZATION_AGGRESSIVE
)


class ComplianceAuditor:
    """Audits data for GDPR/RGPD compliance issues."""
    
    def __init__(self):
        self.anonymizer = TextAnonymizer(AnonymizationConfig(aggressive=True))
        self.violations = []
        
    def audit_text(self, text: str, text_id: str = "unknown") -> Dict:
        """
        Audit a single text for sensitive information.
        
        Returns dict with:
        - has_violations: bool
        - violations: List[dict] with type, text, position
        - risk_level: 'low', 'medium', 'high', 'critical'
        """
        violations = []
        
        # Check for emails
        if self.anonymizer.email_pattern.search(text):
            matches = self.anonymizer.email_pattern.finditer(text)
            for match in matches:
                violations.append({
                    'type': 'EMAIL',
                    'text': match.group(),
                    'position': match.span(),
                    'risk': 'high'
                })
        
        # Check for phone numbers
        for pattern in self.anonymizer.phone_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                violations.append({
                    'type': 'PHONE',
                    'text': match.group(),
                    'position': match.span(),
                    'risk': 'high'
                })
        
        # Check for addresses
        if self.anonymizer.postal_pattern.search(text):
            matches = self.anonymizer.postal_pattern.finditer(text)
            for match in matches:
                violations.append({
                    'type': 'ADDRESS',
                    'text': match.group(),
                    'position': match.span(),
                    'risk': 'medium'
                })
        
        # Check for credit cards
        if self.anonymizer.credit_card_pattern.search(text):
            violations.append({
                'type': 'CREDIT_CARD',
                'text': '[REDACTED]',
                'position': (0, 0),
                'risk': 'critical'
            })
        
        # Check for IBAN
        if self.anonymizer.iban_pattern.search(text):
            violations.append({
                'type': 'IBAN',
                'text': '[REDACTED]',
                'position': (0, 0),
                'risk': 'critical'
            })
        
        # ---------------------------------------------------------------
        # GDPR Article 9 — all special-category sensitive data
        # Uses the compiled patterns from anonymize.py
        # ---------------------------------------------------------------
        art9_findings = self.anonymizer.detect_article9(text)
        seen_categories: set = set()
        for finding in art9_findings:
            cat = finding["category"]
            if cat not in seen_categories:
                seen_categories.add(cat)
                violations.append({
                    'type': cat,
                    'text': finding["matched"],
                    'position': finding["span"],
                    'risk': 'critical'  # All Article 9 data is critical
                })
        
        # Determine overall risk level
        if not violations:
            risk_level = 'compliant'
        elif any(v['risk'] == 'critical' for v in violations):
            risk_level = 'critical'
        elif any(v['risk'] == 'high' for v in violations):
            risk_level = 'high'
        else:
            risk_level = 'medium'
        
        return {
            'text_id': text_id,
            'has_violations': len(violations) > 0,
            'violations': violations,
            'violation_count': len(violations),
            'risk_level': risk_level
        }
    
    def audit_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> Dict:
        """Audit an entire dataset."""
        logger.info(f"Auditing {len(df)} records for GDPR/RGPD compliance...")
        
        results = []
        risk_stats = defaultdict(int)
        violation_types = defaultdict(int)
        
        for idx, row in df.iterrows():
            text = row[text_column]
            text_id = row.get('note_id', row.get('client_id', f'row_{idx}'))
            
            audit_result = self.audit_text(text, text_id)
            results.append(audit_result)
            
            risk_stats[audit_result['risk_level']] += 1
            
            for violation in audit_result['violations']:
                violation_types[violation['type']] += 1
        
        total_violations = sum(r['violation_count'] for r in results)
        records_with_violations = sum(1 for r in results if r['has_violations'])
        
        return {
            'total_records': len(df),
            'records_with_violations': records_with_violations,
            'compliance_rate': (len(df) - records_with_violations) / len(df) * 100,
            'total_violations': total_violations,
            'risk_distribution': dict(risk_stats),
            'violation_types': dict(violation_types),
            'detailed_results': results
        }


def audit_command():
    """Audit current data for GDPR/RGPD compliance."""
    logger.info("=" * 80)
    logger.info("GDPR/RGPD COMPLIANCE AUDIT")
    logger.info("=" * 80)
    
    auditor = ComplianceAuditor()
    
    # Check if processed data exists
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    
    if not notes_path.exists():
        logger.error(f"No data found at {notes_path}")
        logger.error("Please run the pipeline first to generate data.")
        sys.exit(1)
    
    # Load data
    logger.info(f"\nLoading data from {notes_path}...")
    df = pd.read_parquet(notes_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Run audit
    logger.info("\nScanning for sensitive personal information...")
    logger.info(f"Anonymization {'ENABLED' if ENABLE_ANONYMIZATION else 'DISABLED'}")
    logger.info(f"Aggressive mode: {'YES' if ANONYMIZATION_AGGRESSIVE else 'NO'}")
    
    audit_results = auditor.audit_dataset(df, text_column='text')
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("AUDIT RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"  Total records audited: {audit_results['total_records']}")
    logger.info(f"  Records with violations: {audit_results['records_with_violations']}")
    logger.info(f"  Compliance rate: {audit_results['compliance_rate']:.2f}%")
    logger.info(f"  Total violations found: {audit_results['total_violations']}")
    
    logger.info(f"\n⚠️  Risk Distribution:")
    for risk_level, count in audit_results['risk_distribution'].items():
        emoji = {
            'compliant': '✅',
            'medium': '⚠️ ',
            'high': '🔴',
            'critical': '🚨'
        }.get(risk_level, '  ')
        logger.info(f"  {emoji} {risk_level.upper()}: {count} records")
    
    logger.info(f"\n📋 Violation Types:")
    for vtype, count in sorted(audit_results['violation_types'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {vtype}: {count} occurrences")
    
    # Show sample violations
    violations_found = [r for r in audit_results['detailed_results'] if r['has_violations']]
    if violations_found:
        logger.info(f"\n🔍 Sample Violations (showing first 5):")
        for i, result in enumerate(violations_found[:5], 1):
            logger.info(f"\n  {i}. Record ID: {result['text_id']}")
            logger.info(f"     Risk Level: {result['risk_level'].upper()}")
            for v in result['violations'][:3]:  # Max 3 per record
                logger.info(f"     - {v['type']}: {v.get('text', '[REDACTED]')}")
    
    # Save detailed report
    report_path = DATA_OUTPUTS / f"compliance_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 Detailed report saved to: {report_path}")
    
    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    
    if audit_results['compliance_rate'] < 100:
        logger.info("\n⚠️  ACTION REQUIRED:")
        logger.info("  1. Enable anonymization: Set ENABLE_ANONYMIZATION=true")
        logger.info("  2. Re-run the pipeline to anonymize sensitive data")
        logger.info("  3. Review and update anonymization patterns if needed")
        logger.info("  4. Implement data retention policies")
    else:
        logger.info("\n✅ COMPLIANT:")
        logger.info("  All records are properly anonymized.")
        logger.info("  Continue monitoring new data ingestion.")
    
    logger.info("=" * 80)


def test_command(text: str):
    """Test anonymization on sample text."""
    logger.info("=" * 80)
    logger.info("ANONYMIZATION TEST")
    logger.info("=" * 80)
    
    logger.info(f"\nOriginal text:")
    logger.info(f"  {text}")
    
    # Test with different configurations
    configs = [
        ("Conservative", AnonymizationConfig(aggressive=False)),
        ("Aggressive", AnonymizationConfig(aggressive=True)),
    ]
    
    for config_name, config in configs:
        anonymizer = TextAnonymizer(config)
        anonymized = anonymizer.anonymize(text)
        
        logger.info(f"\n{config_name} anonymization:")
        logger.info(f"  {anonymized}")
    
    # Audit the original text
    auditor = ComplianceAuditor()
    audit_result = auditor.audit_text(text)
    
    logger.info(f"\n📊 Detected violations:")
    if audit_result['has_violations']:
        for v in audit_result['violations']:
            logger.info(f"  - {v['type']}: {v.get('text', '[REDACTED]')} (risk: {v['risk']})")
    else:
        logger.info("  ✅ No violations detected")
    
    logger.info("=" * 80)


def config_command():
    """Display current compliance configuration."""
    logger.info("=" * 80)
    logger.info("GDPR/RGPD COMPLIANCE CONFIGURATION")
    logger.info("=" * 80)
    
    logger.info(f"\n🔒 Anonymization Settings:")
    logger.info(f"  Enabled: {'YES ✅' if ENABLE_ANONYMIZATION else 'NO ❌'}")
    logger.info(f"  Aggressive mode: {'YES' if ANONYMIZATION_AGGRESSIVE else 'NO'}")
    
    logger.info(f"\n📁 Data Paths:")
    logger.info(f"  Input: {DATA_INPUT}")
    logger.info(f"  Processed: {DATA_PROCESSED}")
    logger.info(f"  Outputs: {DATA_OUTPUTS}")
    
    config = AnonymizationConfig(aggressive=ANONYMIZATION_AGGRESSIVE)
    
    logger.info(f"\n🎯 Protected Data Types:")
    logger.info(f"  Names: {'YES' if config.redact_names else 'NO'}")
    logger.info(f"  Emails: {'YES' if config.redact_emails else 'NO'}")
    logger.info(f"  Phones: {'YES' if config.redact_phones else 'NO'}")
    logger.info(f"  Addresses: {'YES' if config.redact_addresses else 'NO'}")
    logger.info(f"  Credit Cards: {'YES' if config.redact_credit_cards else 'NO'}")
    logger.info(f"  Bank Accounts: {'YES' if config.redact_bank_accounts else 'NO'}")
    logger.info(f"  ID Numbers: {'YES' if config.redact_id_numbers else 'NO'}")
    logger.info(f"  Dates of Birth: {'YES' if config.redact_dates_of_birth else 'NO'}")
    logger.info(f"  IP Addresses: {'YES' if config.redact_ip_addresses else 'NO'}")
    
    logger.info(f"\n⚙️  To modify configuration:")
    logger.info(f"  export ENABLE_ANONYMIZATION=true")
    logger.info(f"  export ANONYMIZATION_AGGRESSIVE=true")
    
    logger.info("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GDPR/RGPD Compliance Tool - Ensure privacy compliance in LVMH pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Audit command
    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit current data for GDPR/RGPD compliance violations"
    )
    
    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test anonymization on sample text"
    )
    test_parser.add_argument(
        "text",
        type=str,
        help="Text to test anonymization on"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Display current compliance configuration"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "audit":
        audit_command()
    elif args.command == "test":
        test_command(args.text)
    elif args.command == "config":
        config_command()


if __name__ == "__main__":
    main()
