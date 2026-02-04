# RGPD/GDPR Anonymization

## Overview

This module ensures **RGPD** (Règlement Général sur la Protection des Données) and **GDPR** (General Data Protection Regulation) compliance by automatically detecting and anonymizing sensitive personal information (PII) in client transcriptions before any processing occurs.

## What Gets Anonymized

The anonymization module detects and redacts the following types of sensitive information:

### Personal Identifiers
- **Names**: Detects honorifics (M., Mme, Dr., etc.) followed by names
  - Example: `M. Jean Dupont` → `[PERSON]`
- **Email addresses**: Standard email format
  - Example: `jean.dupont@example.com` → `[EMAIL]`

### Contact Information
- **Phone numbers**: French and international formats
  - `06 12 34 56 78` → `[PHONE]`
  - `+33 6 12 34 56 78` → `[PHONE]`
  - `(01) 234-5678` → `[PHONE]`
- **Postal addresses**: Street addresses and postal codes
  - `123 rue de la Paix, 75001 Paris` → `[ADDRESS], [POSTAL_CODE] [CITY]`

### Financial Information
- **Credit card numbers**: 16-digit card numbers
  - `4532 1234 5678 9010` → `[CREDIT_CARD]`
- **IBAN**: European bank account numbers
  - `FR76 3000 6000 0112 3456 7890 189` → `[IBAN]`

### Government IDs
- **ID card numbers**: French Carte Nationale d'Identité
  - `123456789012` → `[ID_NUMBER]`
- **Social security numbers**: Various formats
  - Detected and replaced with `[SSN]`

### Other Sensitive Data
- **Dates of birth**: Birth date mentions
  - `Né le 15/03/1985` → `[DATE_OF_BIRTH]`
- **IP addresses**: IPv4 addresses
  - `192.168.1.1` → `[IP_ADDRESS]`

## Configuration

### Environment Variables

Control anonymization behavior via environment variables or `.env` file:

```bash
# Enable/disable anonymization (default: enabled)
ENABLE_ANONYMIZATION=true

# Aggressiveness level (default: false)
# false = conservative (fewer false positives, may miss some names)
# true = aggressive (catches more names, may have false positives)
ANONYMIZATION_AGGRESSIVE=false
```

### Programmatic Configuration

For more granular control, use `AnonymizationConfig`:

```python
from server.privacy import anonymize_text, AnonymizationConfig

# Custom configuration
config = AnonymizationConfig(
    redact_names=True,
    redact_emails=True,
    redact_phones=True,
    redact_addresses=True,
    redact_credit_cards=True,
    redact_bank_accounts=True,
    redact_id_numbers=True,
    redact_dates_of_birth=True,
    redact_ip_addresses=True,
    placeholder_style="[TYPE]",  # Options: "[TYPE]", "[***]", ""
    aggressive=False
)

clean_text = anonymize_text(original_text, config)
```

## Usage

### Automatic Integration

Anonymization is **automatically applied** during the ingest stage when the pipeline runs:

```bash
# Run pipeline with anonymization (default)
python -m server.run_all

# Run pipeline without anonymization
ENABLE_ANONYMIZATION=false python -m server.run_all
```

### Standalone Usage

You can also use the anonymization module independently:

```python
from server.privacy import anonymize_text

# Simple usage with defaults
text = "M. Jean Dupont (jean@example.com) habite au 123 rue de la Paix."
anonymized = anonymize_text(text)
print(anonymized)
# Output: "[PERSON] ([EMAIL]) habite au [ADDRESS]."
```

### Testing

Test the anonymization module:

```bash
python -m server.privacy.anonymize
```

## Examples

### Before and After

| Original | Anonymized |
|----------|-----------|
| `M. Jean Dupont habite à Paris` | `[PERSON] habite à Paris` |
| `Contact: marie@example.com` | `Contact: [EMAIL]` |
| `Téléphone: 06 12 34 56 78` | `Téléphone: [PHONE]` |
| `123 avenue des Champs-Élysées, 75008 Paris` | `[ADDRESS], [POSTAL_CODE] [CITY]` |
| `Carte: 4532 1234 5678 9010` | `Carte: [CREDIT_CARD]` |
| `IBAN: FR76 3000 6000 0112 3456 7890 189` | `IBAN: [IBAN]` |

### Real-World Example

**Original transcription:**
```
Rendez-vous avec M. Jean Dupont (jean.dupont@gmail.com), client VIP.
Il habite au 45 boulevard Saint-Germain, 75005 Paris.
Téléphone: +33 6 12 34 56 78
Carte bancaire: 4532 1234 5678 9010
Souhaite commander la nouvelle collection joaillerie pour un événement corporate.
```

**Anonymized transcription:**
```
Rendez-vous avec [PERSON] ([EMAIL]), client VIP.
Il habite au [ADDRESS], [POSTAL_CODE] [CITY].
Téléphone: [PHONE]
Carte bancaire: [CREDIT_CARD]
Souhaite commander la nouvelle collection joaillerie pour un événement corporate.
```

**Key insights are preserved:**
- Client is VIP
- Interest in "joaillerie" (jewelry)
- "corporate" context
- All PII removed ✅

## How It Works

### Pipeline Integration

```
1. CSV Input (with raw transcriptions)
   ↓
2. Ingest Stage
   ├─ Load CSV
   ├─ Validate columns
   ├─ Clean text
   └─ **Anonymize** ← Happens here
   ↓
3. Candidate Extraction (on anonymized text)
   ↓
4. Rest of pipeline (all work on anonymized data)
```

### Pattern Matching

The anonymizer uses regex patterns and linguistic rules to detect:
- French honorifics (M., Mme, Dr., etc.)
- Capitalized name sequences
- Standard formats for emails, phones, addresses
- French postal codes (5 digits + city name)
- Credit card patterns
- IBAN formats

### Preservation

The anonymizer is designed to:
- **Preserve semantic content**: Product names, concepts, and intents remain intact
- **Maintain structure**: Sentence structure and readability are preserved
- **Be deterministic**: Same input always produces same output
- **Be multilingual**: Works with French, English, Italian, Spanish, German

## Data Protection

### What's Protected

✅ **Protected by anonymization:**
- Personal names
- Contact details (email, phone, address)
- Financial information (cards, IBANs)
- Government IDs
- Birthdates

✅ **Never leaves your infrastructure:**
- All processing is done locally
- No data sent to external services
- No LLMs involved in anonymization

### What's Preserved

✅ **Preserved for analysis:**
- Product mentions (watches, jewelry, perfumes)
- Interests and preferences
- Languages and styles
- Business context (VIP, corporate, budget)
- Intent signals (purchasing, browsing)

## Compliance Notes

### RGPD Article 32 - Security of Processing

The anonymization module helps satisfy RGPD Article 32 requirements for:
- **Pseudonymization**: PII is replaced with placeholders
- **Confidentiality**: Sensitive data is not exposed in outputs
- **Data minimization**: Only necessary information is retained

### Legal Considerations

⚠️ **Important Notes:**
1. This module provides **technical anonymization**, not legal advice
2. Consult with your Data Protection Officer (DPO) for full compliance
3. Consider additional measures:
   - Data access logging
   - Retention policies
   - User consent management
4. Test thoroughly with your specific data types

### Limitations

- **Not 100% accurate**: Some edge cases may be missed
- **Context-dependent**: May not catch unconventional PII formats
- **False positives possible**: Especially in aggressive mode
- **Language-specific**: Optimized for French/European patterns

### Recommendations

1. **Test with real data samples** before production use
2. **Review outputs** to ensure important concepts aren't over-redacted
3. **Adjust aggressiveness** based on your data and requirements
4. **Document** your anonymization approach for compliance audits
5. **Regularly update** patterns as new PII types emerge

## API Reference

### `anonymize_text(text, config=None)`

Main function to anonymize text.

**Parameters:**
- `text` (str): Input text to anonymize
- `config` (AnonymizationConfig, optional): Configuration object

**Returns:**
- str: Anonymized text

### `AnonymizationConfig`

Configuration dataclass for anonymization behavior.

**Attributes:**
- `redact_names` (bool): Anonymize personal names
- `redact_emails` (bool): Anonymize email addresses
- `redact_phones` (bool): Anonymize phone numbers
- `redact_addresses` (bool): Anonymize postal addresses
- `redact_credit_cards` (bool): Anonymize credit card numbers
- `redact_bank_accounts` (bool): Anonymize IBAN/bank accounts
- `redact_id_numbers` (bool): Anonymize ID card numbers
- `redact_dates_of_birth` (bool): Anonymize birthdates
- `redact_ip_addresses` (bool): Anonymize IP addresses
- `placeholder_style` (str): Style of placeholders (`"[TYPE]"`, `"[***]"`, `""`)
- `aggressive` (bool): Use aggressive matching

## Support

For issues or questions about RGPD compliance:
1. Review this documentation
2. Test with sample data
3. Consult with your legal/compliance team
4. Adjust configuration as needed

---

**Last updated:** February 2026  
**Module:** `server/privacy/`
