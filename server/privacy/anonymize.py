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
- GDPR Article 9 special-category data:
    - Health / medical mentions
    - Sexual orientation
    - Religious / philosophical beliefs
    - Political opinions
    - Ethnic / racial origin
    - Trade-union membership
    - Criminal / judicial history
    - Financial difficulties (over-indebtedness, debt collection)
    - Conflictual family situations (divorce, custody disputes)
    - Physical appearance comments

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
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# GDPR Article 9 — sensitive-category keyword dictionaries (FR + EN)
# Each key maps to a list of regex patterns (case-insensitive) that identify
# mentions of special-category personal data.
# ---------------------------------------------------------------------------
ARTICLE9_PATTERNS: Dict[str, List[str]] = {
    # --- Health / Medical -----------------------------------------------
    "HEALTH_DATA": [
        # diseases, conditions, symptoms
        r"\b(?:diab[eè]t\w*|cancer\w*|tumeur\w*|tumor\w*|asthm\w*|épileps\w*|epileps\w*|VIH|HIV|sida|aids)\b",
        r"\b(?:maladie\w*|disease\w*|illness\w*|patholog\w*|symptôm\w*|symptom\w*|diagnostic\w*|diagnosis)\b",
        r"\b(?:allergi\w*|allergy|intol[eé]ran\w*|handicap\w*|disabilit\w*|infirmit\w*)\b",
        r"\b(?:m[eé]dicament\w*|medication\w*|medicine\w*|traitement\s+médical\w*|ordonnance\w*|prescription\w*)\b",
        r"\b(?:hospitalis\w*|chirurgi\w*|surgery|opéra[tion]+\s+(?:médical|chirurgical)\w*)\b",
        r"\b(?:dépression\w*|depression\w*|anxi[eé]t\w*|anxiety|psychiatr\w*|psycholog\w*|th[eé]rapi\w*)\b",
        r"\b(?:grossesse\w*|pregnancy|enceinte\w*|pregnant|fausse\s+couche|miscarriage)\b",
        r"\b(?:troubles?\s+(?:alimentaire|eating)|anorexi\w*|boulimi\w*|bulimi\w*)\b",
    ],
    # --- Sexual orientation / sex life -----------------------------------
    "SEXUAL_ORIENTATION": [
        r"\b(?:homosexu|hétérosexu|heterosexu|bisexu|pansexu|asexu)\w*\b",
        r"\b(?:orientation\s+sexuelle|sexual\s+orientation|identit[eé]\s+(?:de\s+genre|sexuelle))\b",
        r"\b(?:LGBT|LGBTQ|queer|transgenre|transgender|non[- ]binaire|non[- ]binary)\b",
        r"\b(?:gay|lesbien\w*|lesbian)\b",
        r"\b(?:vie\s+sexuelle|sex\s+life|partner\s+sexuel)\b",
    ],
    # --- Religious / philosophical beliefs --------------------------------
    "RELIGIOUS_BELIEF": [
        r"\b(?:musulman\w*|muslim\w*|islam\w*|chrétien\w*|christian\w*|catholi\w*|protestant\w*|évangéli\w*|evangelical\w*)\b",
        r"\b(?:juif|juive|jewish|judaï\w*|bouddhist\w*|buddhist\w*|hindou\w*|hindu\w*|sikh\w*|athé\w*|atheist\w*|agnosti\w*)\b",
        r"\b(?:religion\w*|religieu\w*|religious|croyance\w*|belief\w*|culte\w*|worship\w*|pratiquant\w*|practicing)\b",
        r"\b(?:ramadan|shabbat|carême|lent|halal|casher|kosher|prière\w*|prayer\w*|mosquée\w*|mosque\w*|synagogue\w*|temple\w*|église\w*|church\w*)\b",
    ],
    # --- Political opinions -----------------------------------------------
    "POLITICAL_OPINION": [
        r"\b(?:opinion\s+politique|political\s+opinion|convictions?\s+politiques?)\b",
        r"\b(?:parti\s+politique|political\s+party|militant\w*|activism\w*|activiste\w*|activist\w*)\b",
        r"\b(?:extrême\s+(?:droite|gauche)|far[- ](?:right|left)|syndical\w*|trade\s+union)\b",
        r"\b(?:adhérent\w*|member\s+of\s+party|vote\s+pour|voted\s+for|sympathisant\w*|supporter\w*)\b",
        r"\b(?:grève\w*|strike\w*|manifesta[tion]+\s+politique|political\s+protest)\b",
    ],
    # --- Ethnic / racial origin -------------------------------------------
    "ETHNIC_ORIGIN": [
        r"\b(?:origine\s+(?:ethnique|raciale)|ethnic\s+(?:origin|background)|racial\s+origin)\b",
        r"\b(?:couleur\s+de\s+peau|skin\s+colo[u]?r)\b",
    ],
    # --- Trade-union membership -------------------------------------------
    "TRADE_UNION": [
        r"\b(?:syndicat\w*|trade\s+union|union\s+member|adhésion\s+syndicale|syndiqué\w*)\b",
        r"\b(?:CGT|CFDT|UNSA|comité\s+d'entreprise|works\s+council)\b",
    ],
    # --- Criminal / judicial history -------------------------------------
    "CRIMINAL_RECORD": [
        r"\b(?:casier\s+judiciaire|criminal\s+record|condamna[tion]+\w*)\b",
        r"\b(?:infraction\w*|offense\w*|délit\w*|misdemeanor\w*|felony|crime\b|criminel\w*)\b",
        r"\b(?:garde\s+à\s+vue|détention\w*|detention\w*|prison\w*|emprisonnem\w*|incarcér\w*)\b",
        r"\b(?:procès|jugement\w*|judgment\w*|court\s+(?:case|hearing))\b",
    ],
    # --- Financial difficulties ------------------------------------------
    "FINANCIAL_DIFFICULTY": [
        r"\b(?:surendett\w*|over[- ]indebt\w*|dette[s]?|debt[s]?\b|insolvab\w*|insolven\w*)\b",
        r"\b(?:faillite\w*|bankruptcy|redressement\s+judiciaire|liquidation\s+judiciaire)\b",
        r"\b(?:interdit\s+bancaire|banking\s+ban|fichier?\s+(?:Banque\s+de\s+France|FICP))\b",
        r"\b(?:saisie\w*|seizure\w*|huissier\w*|bailiff\w*|recouvrement\s+de\s+(?:dette|créance))\b",
    ],
    # --- Conflictual family situations ------------------------------------
    "FAMILY_CONFLICT": [
        r"\b(?:divorce\w*|séparation\s+(?:judiciaire|conjugale)|legal\s+separation)\b",
        r"\b(?:garde\s+(?:des?\s+enfants?|alternée|exclusive)|child\s+custody)\b",
        r"\b(?:pension\s+alimentaire|alimony|child\s+support)\b",
        r"\b(?:violence\s+(?:conjugale|domestique|familiale)|domestic\s+(?:violence|abuse))\b",
        r"\b(?:ordonnance\s+de\s+protection|restraining\s+order|harcèlement\s+(?:conjugal|familial))\b",
    ],
    # --- Physical appearance comments ------------------------------------
    "PHYSICAL_APPEARANCE": [
        r"\b(?:physique\s+(?:disgracieu\w*|ingrat\w*|repoussant\w*)|ugly|unattractive)\b",
        r"\b(?:ob[eè]se\w*|obesity|surpoids\w*|overweight|anorexique\w*|maigre\s+(?:excessif|extreme))\b",
        r"\b(?:cicatrice\w*|scarr(?:ed|ing)\b|difformit\w*|deformit\w*|infirmit[eé]\s+(?:visible|physique))\b",
        r"\b(?:commentaire\s+sur\s+(?:le\s+)?physique|body\s+shaming)\b",
    ],
}

# Pre-compile all Article 9 patterns for performance
_COMPILED_ARTICLE9: Dict[str, List[re.Pattern]] = {}
for _cat, _patterns in ARTICLE9_PATTERNS.items():
    _COMPILED_ARTICLE9[_cat] = [re.compile(p, re.IGNORECASE) for p in _patterns]


@dataclass
class AnonymizationConfig:
    """Configuration for anonymization behavior."""
    
    # What to anonymize — PII
    redact_names: bool = True
    redact_emails: bool = True
    redact_phones: bool = True
    redact_addresses: bool = True
    redact_credit_cards: bool = True
    redact_bank_accounts: bool = True
    redact_id_numbers: bool = True
    redact_dates_of_birth: bool = True
    redact_ip_addresses: bool = True
    
    # GDPR Article 9 — special-category data
    redact_article9: bool = True  # Flag/redact sensitive-category mentions
    article9_mode: str = "flag"   # "flag" = [SENSITIVE:TYPE], "redact" = remove, "log" = detect only
    
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
        
        # Lazy-loaded ML safety-net detector (only loaded once, on first use)
        self._ml_detector: Optional[object] = None
        self._ml_detector_checked = False
        
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

    # ------------------------------------------------------------------
    # GDPR Article 9 — special-category sensitive data
    # ------------------------------------------------------------------

    # Context words that neutralise a match (product / geographic context)
    _ART9_FALSE_POS = re.compile(
        r"\b(?:tissu|motif|bijou|style|imprimé|fabric|print|pattern|"
        r"Amérique|America|hémisphère|hemisphere|du\s+Sud|d'Asie|"
        r"tribunal\s+(?:supérieur|administratif|de\s+commerce)|"
        r"juge[za]?\s+tribunal|"
        r"convictions?\s+(?:éthique|morale|religieu|personnelle|ethical|moral|personal))\b",
        re.IGNORECASE,
    )

    def detect_article9(self, text: str) -> List[Dict]:
        """
        Detect GDPR Article 9 sensitive-category data in text.

        Returns a list of dicts:
          [{"category": "HEALTH_DATA", "matched": "allergie", "span": (12, 20)}, ...]
        """
        findings: List[Dict] = []
        for category, patterns in _COMPILED_ARTICLE9.items():
            for pat in patterns:
                for m in pat.finditer(text):
                    # Grab surrounding context (40 chars each side) for FP check
                    ctx_start = max(0, m.start() - 40)
                    ctx_end = min(len(text), m.end() + 40)
                    context = text[ctx_start:ctx_end]
                    if self._ART9_FALSE_POS.search(context):
                        continue  # skip false positive
                    findings.append({
                        "category": category,
                        "matched": m.group(),
                        "span": m.span(),
                    })
        return findings

    def _anonymize_article9(self, text: str) -> str:
        """Apply Article 9 handling based on config.article9_mode."""
        if not self.config.redact_article9:
            return text

        findings = self.detect_article9(text)
        if not findings:
            return text

        mode = self.config.article9_mode

        if mode == "log":
            # Detection only — no text modification
            return text

        # Sort by position descending so replacements don't shift offsets
        findings.sort(key=lambda f: f["span"][0], reverse=True)

        # De-duplicate overlapping spans
        seen_spans: set = set()
        for f in findings:
            span = f["span"]
            if any(span[0] < e and span[1] > s for (s, e) in seen_spans):
                continue
            seen_spans.add(span)

            if mode == "flag":
                placeholder = f"[SENSITIVE:{f['category']}]"
            else:  # "redact"
                placeholder = self._get_placeholder(f"SENSITIVE_{f['category']}")

            text = text[:span[0]] + placeholder + text[span[1]:]

        return text

    # ------------------------------------------------------------------
    # ML Safety-Net — catches sensitive words that regex missed
    # ------------------------------------------------------------------

    def _get_ml_detector(self):
        """Lazy-load the ML sensitive-word detector (singleton)."""
        if not self._ml_detector_checked:
            self._ml_detector_checked = True
            try:
                from server.privacy.sensitive_model import SensitiveWordDetector
                det = SensitiveWordDetector()
                if det.available:
                    self._ml_detector = det
            except Exception:
                pass  # Model not trained or import error — silently skip
        return self._ml_detector

    def _ml_safety_net(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Run the ML NER model on text to catch sensitive words the regex missed.

        Only flags/redacts tokens that were NOT already caught by regex.
        Returns (possibly_modified_text, ml_findings).
        """
        if not self.config.redact_article9:
            return text, []

        detector = self._get_ml_detector()
        if detector is None:
            return text, []

        ml_hits = detector.predict(text)
        if not ml_hits:
            return text, []

        # Get regex findings so we can skip already-caught spans
        regex_findings = self.detect_article9(text)
        regex_spans = {(f["span"][0], f["span"][1]) for f in regex_findings}

        # Only keep ML hits that don't overlap with regex hits
        novel_hits: List[Dict] = []
        for hit in ml_hits:
            s, e = hit["start"], hit["end"]
            overlaps = any(s < re and e > rs for (rs, re) in regex_spans)
            if not overlaps:
                novel_hits.append(hit)

        if not novel_hits:
            return text, []

        mode = self.config.article9_mode

        if mode == "log":
            # Just return findings, don't modify text
            return text, novel_hits

        # Apply replacements (reverse order to preserve offsets)
        novel_hits.sort(key=lambda h: h["start"], reverse=True)
        for hit in novel_hits:
            if mode == "flag":
                placeholder = f"[SENSITIVE:{hit['label']}]"
            else:
                placeholder = self._get_placeholder(f"SENSITIVE_{hit['label']}")
            text = text[: hit["start"]] + placeholder + text[hit["end"] :]

        return text, novel_hits

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
        text = self._anonymize_article9(text)
        
        # ML safety net — final pass to catch anything regex missed
        text, _ml_findings = self._ml_safety_net(text)
        
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
