"""
RGPD/GDPR Anonymization Module

This module detects and anonymizes sensitive personal information (PII) in text
to ensure compliance with RGPD (Règlement Général sur la Protection des Données)
and GDPR (General Data Protection Regulation).

Detects and redacts:
- Names (French, English, and other common names)
- Email addresses
- Phone numbers (French and international formats)
- Postal addresses
- Credit card numbers
- IBAN/Bank account numbers
- French Carte Nationale d'Identité numbers
- Social security numbers
- IP addresses
- Dates of birth

Usage:
    from server.privacy import anonymize_text, AnonymizationConfig
    
    # Basic usage
    clean_text = anonymize_text("M. Jean Dupont habite au 123 rue de la Paix, 75001 Paris.")
    # Output: "[PERSON] habite au [ADDRESS], [POSTAL_CODE] [CITY]."
    
    # Custom configuration
    config = AnonymizationConfig(
        redact_names=True,
        redact_emails=True,
        redact_phones=True,
        redact_addresses=True,
        placeholder_style="[TYPE]"  # or "[***]" or ""
    )
    clean_text = anonymize_text(text, config)
"""
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnonymizationConfig:
    """Configuration for anonymization behavior."""
    
    # What to anonymize
    redact_names: bool = True
    redact_emails: bool = True
    redact_phones: bool = True
    redact_addresses: bool = True
    redact_credit_cards: bool = True
    redact_bank_accounts: bool = True
    redact_id_numbers: bool = True
    redact_dates_of_birth: bool = True
    redact_ip_addresses: bool = True
    
    # How to anonymize
    placeholder_style: str = "[TYPE]"  # "[TYPE]", "[***]", "" (remove), or custom
    preserve_structure: bool = True  # Keep sentence structure intact
    
    # Sensitivity level
    aggressive: bool = False  # If True, redact more aggressively (may have false positives)


class TextAnonymizer:
    """Main anonymization class with pattern matching and replacement logic."""
    
    def __init__(self, config: Optional[AnonymizationConfig] = None):
        """Initialize anonymizer with configuration."""
        self.config = config or AnonymizationConfig()
        
        # Common French honorifics and titles
        self.honorifics = [
            r'\b(?:M\.|Mr\.|Monsieur|Mme|Madame|Mlle|Mademoiselle|Dr\.|Docteur|Prof\.|Professeur)\b',
        ]
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for sensitive data detection."""
        
        # Email pattern (standard)
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        # Phone patterns (French and international)
        self.phone_patterns = [
            # French formats: 06 12 34 56 78, 06.12.34.56.78, 06-12-34-56-78, 0612345678
            re.compile(r'\b0[1-9](?:[\s.-]?\d{2}){4}\b'),
            # International: +33 6 12 34 56 78, +33612345678
            re.compile(r'\+\d{1,3}[\s.-]?\d{1,4}(?:[\s.-]?\d{2,4}){2,4}\b'),
            # (XX) XXX-XXXX format
            re.compile(r'\(\d{2,3}\)[\s.-]?\d{3}[\s.-]?\d{4}\b'),
        ]
        
        # French postal code + city pattern
        self.postal_pattern = re.compile(
            r'\b\d{5}\s+[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ][a-zàâäéèêëïîôùûüÿç]+(?:[\s-][A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ][a-zàâäéèêëïîôùûüÿç]+)*\b',
            re.IGNORECASE
        )
        
        # Street address patterns (French)
        self.address_patterns = [
            # "123 rue de la Paix", "5 avenue des Champs-Élysées"
            re.compile(r'\b\d+\s+(?:rue|avenue|av\.|boulevard|bd\.|place|impasse|allée|chemin|quai|cours)\s+[A-Za-zÀ-ÿ\s-]+(?=\s*,|\s*\d{5}|\.|$)', re.IGNORECASE),
            # "Bâtiment A", "Appartement 5"
            re.compile(r'\b(?:bâtiment|batiment|bat\.|appartement|appt\.|étage|escalier)\s+[A-Za-z0-9]+\b', re.IGNORECASE),
        ]
        
        # Credit card pattern (simplified)
        self.credit_card_pattern = re.compile(
            r'\b(?:\d{4}[\s-]?){3}\d{4}\b'
        )
        
        # IBAN pattern (European bank account)
        self.iban_pattern = re.compile(
            r'\b[A-Z]{2}\d{2}[\s]?(?:\d{4}[\s]?){3,7}\d{1,4}\b'
        )
        
        # French ID card number (example pattern)
        self.french_id_pattern = re.compile(
            r'\b\d{12}\b'
        )
        
        # Date of birth patterns (DD/MM/YYYY, DD-MM-YYYY, etc.)
        self.dob_patterns = [
            re.compile(r'\b(?:né|née|naissance|né le|née le)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b'),  # Conservative: only 1900-2099
        ]
        
        # IP address pattern
        self.ip_pattern = re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        )
    
    def _get_placeholder(self, entity_type: str) -> str:
        """Generate placeholder text based on configuration."""
        if self.config.placeholder_style == "[TYPE]":
            return f"[{entity_type}]"
        elif self.config.placeholder_style == "[***]":
            return "[***]"
        elif self.config.placeholder_style == "":
            return ""
        else:
            return self.config.placeholder_style
    
    def _anonymize_emails(self, text: str) -> str:
        """Anonymize email addresses."""
        if not self.config.redact_emails:
            return text
        return self.email_pattern.sub(self._get_placeholder("EMAIL"), text)
    
    def _anonymize_phones(self, text: str) -> str:
        """Anonymize phone numbers."""
        if not self.config.redact_phones:
            return text
        for pattern in self.phone_patterns:
            text = pattern.sub(self._get_placeholder("PHONE"), text)
        return text
    
    def _anonymize_addresses(self, text: str) -> str:
        """Anonymize postal addresses."""
        if not self.config.redact_addresses:
            return text
        
        # Anonymize postal codes + cities
        text = self.postal_pattern.sub(
            f"{self._get_placeholder('POSTAL_CODE')} {self._get_placeholder('CITY')}", 
            text
        )
        
        # Anonymize street addresses
        for pattern in self.address_patterns:
            text = pattern.sub(self._get_placeholder("ADDRESS"), text)
        
        return text
    
    def _anonymize_credit_cards(self, text: str) -> str:
        """Anonymize credit card numbers."""
        if not self.config.redact_credit_cards:
            return text
        return self.credit_card_pattern.sub(self._get_placeholder("CREDIT_CARD"), text)
    
    def _anonymize_bank_accounts(self, text: str) -> str:
        """Anonymize IBAN and bank account numbers."""
        if not self.config.redact_bank_accounts:
            return text
        return self.iban_pattern.sub(self._get_placeholder("IBAN"), text)
    
    def _anonymize_id_numbers(self, text: str) -> str:
        """Anonymize ID card numbers."""
        if not self.config.redact_id_numbers:
            return text
        return self.french_id_pattern.sub(self._get_placeholder("ID_NUMBER"), text)
    
    def _anonymize_dates_of_birth(self, text: str) -> str:
        """Anonymize dates of birth."""
        if not self.config.redact_dates_of_birth:
            return text
        for pattern in self.dob_patterns:
            text = pattern.sub(self._get_placeholder("DATE_OF_BIRTH"), text)
        return text
    
    def _anonymize_ip_addresses(self, text: str) -> str:
        """Anonymize IP addresses."""
        if not self.config.redact_ip_addresses:
            return text
        return self.ip_pattern.sub(self._get_placeholder("IP_ADDRESS"), text)
    
    def _anonymize_names(self, text: str) -> str:
        """
        Anonymize personal names.
        This uses pattern matching for honorifics and capitalized words.
        """
        if not self.config.redact_names:
            return text
        
        # Pattern: Honorific + Capitalized Name(s)
        # e.g., "M. Jean Dupont", "Mme Marie Martin"
        for honorific_pattern in self.honorifics:
            # Match honorific followed by 1-3 capitalized words
            pattern = re.compile(
                f"{honorific_pattern}\\s+([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ][a-zàâäéèêëïîôùûüÿç]+(?:\\s+[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ][a-zàâäéèêëïîôùûüÿç]+){{0,2}})",
                re.IGNORECASE
            )
            text = pattern.sub(self._get_placeholder("PERSON"), text)
        
        # If aggressive mode, also redact sequences of 2-3 capitalized words
        # that might be names without honorifics (higher false positive rate)
        if self.config.aggressive:
            # Match 2-3 consecutive capitalized words not at sentence start
            name_pattern = re.compile(
                r'(?<=[.!?]\s)\b[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ][a-zàâäéèêëïîôùûüÿç]+(?:\s+[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ][a-zàâäéèêëïîôùûüÿç]+){1,2}\b'
            )
            text = name_pattern.sub(self._get_placeholder("PERSON"), text)
        
        return text
    
    def anonymize(self, text: str) -> str:
        """
        Main anonymization method. Applies all configured anonymizations.
        
        Args:
            text: Input text to anonymize
            
        Returns:
            Anonymized text with sensitive information redacted
        """
        if not text or not isinstance(text, str):
            return text
        
        # Apply anonymizations in order
        # (Order matters to avoid pattern conflicts)
        text = self._anonymize_emails(text)
        text = self._anonymize_phones(text)
        text = self._anonymize_credit_cards(text)
        text = self._anonymize_bank_accounts(text)
        text = self._anonymize_id_numbers(text)
        text = self._anonymize_dates_of_birth(text)
        text = self._anonymize_ip_addresses(text)
        text = self._anonymize_addresses(text)
        text = self._anonymize_names(text)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# Convenience function for simple usage
def anonymize_text(text: str, config: Optional[AnonymizationConfig] = None) -> str:
    """
    Anonymize sensitive information in text.
    
    Args:
        text: Input text to anonymize
        config: Optional configuration for anonymization behavior
        
    Returns:
        Anonymized text with PII redacted
        
    Example:
        >>> anonymize_text("M. Jean Dupont, email: jean@example.com, tel: 06 12 34 56 78")
        "[PERSON], email: [EMAIL], tel: [PHONE]"
    """
    anonymizer = TextAnonymizer(config)
    return anonymizer.anonymize(text)


# Quick test
if __name__ == "__main__":
    test_texts = [
        "M. Jean Dupont habite au 123 rue de la Paix, 75001 Paris. Tél: 06 12 34 56 78.",
        "Mme Marie Martin (marie.martin@example.com) souhaite commander.",
        "Client VIP: +33 6 98 76 54 32, IBAN: FR76 3000 6000 0112 3456 7890 189",
        "Carte bancaire: 4532 1234 5678 9010",
        "Née le 15/03/1985 à Lyon."
    ]
    
    print("=== Anonymization Test ===\n")
    for text in test_texts:
        anonymized = anonymize_text(text)
        print(f"Original:    {text}")
        print(f"Anonymized:  {anonymized}")
        print()
