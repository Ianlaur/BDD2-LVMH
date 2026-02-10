"""
RGPD Sensitive-Word Safety-Net Model (PyTorch / Transformers)

A lightweight token-classification model that acts as a **final gate** after the
regex-based Article 9 detection.  While regex catches known patterns, this model
generalises to morphological variants, misspellings, and novel phrasings.

Architecture
~~~~~~~~~~~~
- Base: ``distilbert-base-multilingual-cased``  (134 M params, ~500 MB)
  — multilingual (104 languages), fast inference on CPU/MPS.
- Head: Linear(768 → num_labels)  — BIO token classification.
  Labels: O, B-HEALTH, I-HEALTH, B-RELIGION, I-RELIGION, …
- Training: fine-tuned on synthetic examples generated from
  ``ARTICLE9_PATTERNS`` + luxury-domain negative examples.

Usage
~~~~~
Train::

    python -m server.privacy.sensitive_model train --epochs 6

Predict::

    from server.privacy.sensitive_model import SensitiveWordDetector
    det = SensitiveWordDetector()
    hits = det.predict("Le client souffre d'épilepsie sévère.")
    # [{"token": "épilepsie", "label": "HEALTH_DATA", "score": 0.97, "start": 23, "end": 33}]

Integration::

    The ``TextAnonymizer`` class in ``anonymize.py`` calls ``SensitiveWordDetector``
    as a final pass after regex.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent.parent / "models" / "sensitive_ner"
_LABEL_MAP_FILE = _MODEL_DIR / "label_map.json"

# Article-9 categories → BIO label prefixes
_CATEGORIES = [
    "HEALTH_DATA",
    "SEXUAL_ORIENTATION",
    "RELIGIOUS_BELIEF",
    "POLITICAL_OPINION",
    "ETHNIC_ORIGIN",
    "TRADE_UNION",
    "CRIMINAL_RECORD",
    "FINANCIAL_DIFFICULTY",
    "FAMILY_CONFLICT",
    "PHYSICAL_APPEARANCE",
]

# ---------------------------------------------------------------------------
# Label map (BIO tagging)
# ---------------------------------------------------------------------------

def _build_label_map() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build label ↔ id maps.  O=0, then B-<cat>, I-<cat> for each category."""
    label2id: Dict[str, int] = {"O": 0}
    idx = 1
    for cat in _CATEGORIES:
        label2id[f"B-{cat}"] = idx
        idx += 1
        label2id[f"I-{cat}"] = idx
        idx += 1
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


LABEL2ID, ID2LABEL = _build_label_map()
NUM_LABELS = len(LABEL2ID)

# ---------------------------------------------------------------------------
# Synthetic training-data generator
# ---------------------------------------------------------------------------
# Luxury-domain carrier sentences (multilingual)
_CARRIER_FR = [
    "Le client souhaite un sac {SLOT} pour sa femme.",
    "Elle a mentionné qu'elle est {SLOT} et cherche un cadeau.",
    "Il souffre de {SLOT} depuis plusieurs années.",
    "La cliente est {SLOT} et préfère des matériaux hypoallergéniques.",
    "Discussion sur {SLOT} lors de la visite en boutique.",
    "Le client a indiqué être {SLOT} ce qui influence ses achats.",
    "Rendez-vous avec une personne {SLOT} intéressée par la haute joaillerie.",
    "Notes du vendeur : le client est {SLOT} apparemment.",
    "Conversation autour de {SLOT} pendant l'essayage.",
    "Profil client : {SLOT} selon ses propres termes.",
]

_CARRIER_EN = [
    "The client mentioned having {SLOT} and prefers hypoallergenic materials.",
    "She indicated she is {SLOT} and is looking for a gift.",
    "He suffers from {SLOT} and wants something gentle.",
    "The customer is {SLOT} and is interested in limited editions.",
    "Discussion about {SLOT} during the store visit.",
    "Client profile: {SLOT} per their own words.",
    "Meeting with a {SLOT} person who loves luxury leather goods.",
    "Notes: the client is {SLOT} apparently.",
    "Conversation about {SLOT} during the fitting.",
    "She has been dealing with {SLOT} for years.",
]

# Sensitive seed words per category (used to fill {SLOT})
_SENSITIVE_SEEDS: Dict[str, List[str]] = {
    "HEALTH_DATA": [
        "diabète", "cancer", "asthme", "épilepsie", "VIH", "allergie sévère",
        "dépression", "anxiété", "handicap moteur", "maladie chronique",
        "diabetes", "cancer", "asthma", "epilepsy", "HIV", "severe allergy",
        "depression", "anxiety", "disability", "chronic illness",
        "grossesse", "pregnancy", "troubles alimentaires", "eating disorder",
        "anorexie", "anorexia", "boulimie", "bulimia",
    ],
    "SEXUAL_ORIENTATION": [
        "homosexuel", "hétérosexuel", "bisexuel", "pansexuel",
        "homosexual", "heterosexual", "bisexual", "pansexual",
        "gay", "lesbienne", "lesbian", "LGBT", "queer", "transgenre", "transgender",
        "non-binaire", "non-binary", "orientation sexuelle", "sexual orientation",
    ],
    "RELIGIOUS_BELIEF": [
        "musulman", "chrétien", "catholique", "protestant", "juif", "bouddhiste",
        "muslim", "christian", "catholic", "protestant", "jewish", "buddhist",
        "hindou", "hindu", "athée", "atheist", "pratiquant", "practicing",
        "ramadan", "shabbat", "halal", "casher", "kosher",
    ],
    "POLITICAL_OPINION": [
        "militant syndical", "trade union activist", "extrême droite", "far right",
        "extrême gauche", "far left", "opinion politique", "political opinion",
        "sympathisant", "supporter", "gréviste", "striker",
        "manifestation politique", "political protest",
    ],
    "ETHNIC_ORIGIN": [
        "origine ethnique", "ethnic origin", "origine raciale", "racial origin",
        "couleur de peau", "skin colour",
    ],
    "TRADE_UNION": [
        "syndicat", "trade union", "syndiqué", "union member",
        "CGT", "CFDT", "adhésion syndicale", "comité d'entreprise",
    ],
    "CRIMINAL_RECORD": [
        "casier judiciaire", "criminal record", "condamnation", "conviction",
        "détention", "detention", "prison", "emprisonnement", "incarcération",
        "garde à vue", "procès", "jugement",
    ],
    "FINANCIAL_DIFFICULTY": [
        "surendetté", "over-indebted", "faillite", "bankruptcy",
        "interdit bancaire", "banking ban", "saisie", "seizure",
        "huissier", "bailiff", "dette", "debt",
    ],
    "FAMILY_CONFLICT": [
        "divorce", "séparation judiciaire", "legal separation",
        "garde des enfants", "child custody", "pension alimentaire", "alimony",
        "violence conjugale", "domestic violence", "violence domestique",
    ],
    "PHYSICAL_APPEARANCE": [
        "obèse", "obese", "surpoids", "overweight",
        "cicatrice visible", "visible scar", "difformité", "deformity",
        "body shaming",
    ],
}

# Negative (safe) luxury-domain sentences
_NEGATIVE_SENTENCES = [
    "Le client souhaite un sac Capucines en cuir taurillon noir.",
    "She is looking for a birthday gift, budget around 3000 euros.",
    "Discussion about leather types: veau, agneau, crocodile.",
    "Il préfère le hardware doré rose pour la collection printemps.",
    "The customer wants to try the new Dauphine wallet in Monogram.",
    "Budget élevé, intéressée par la haute joaillerie Boucheron.",
    "Elle cherche une montre Tank Française pour son mari.",
    "Client régulier, collectionne les éditions limitées.",
    "Conversation about travel retail and duty-free preferences.",
    "Il vient pour un entretien de sa malle Louis Vuitton ancienne.",
    "La cliente aime les parfums orientaux et les notes boisées.",
    "Looking for a scarf in cashmere with geometric print.",
    "Discussion sur les tissus ethniques et motifs traditionnels.",
    "She mentioned her ethical convictions about sustainable fashion.",
    "Le client habite en Amérique du Sud et voyage souvent en Europe.",
    "Conversation about the new South Beach collection.",
    "He is interested in the tribunal superieur limited edition watch.",
    "La cliente a des convictions éthiques fortes sur le cuir végan.",
    "Rendez-vous pris pour essayage robe cocktail soirée gala.",
    "Il cherche un portefeuille compact pour voyages d'affaires.",
]


def _generate_training_examples(
    n_positive: int = 2000,
    n_negative: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate synthetic BIO-tagged training examples.

    Returns list of {"tokens": [...], "labels": [...]}
    """
    rng = random.Random(seed)
    examples: List[Dict] = []

    carriers = _CARRIER_FR + _CARRIER_EN

    # --- Positive examples ---
    for _ in range(n_positive):
        cat = rng.choice(_CATEGORIES)
        seeds = _SENSITIVE_SEEDS[cat]
        phrase = rng.choice(seeds)
        carrier = rng.choice(carriers)
        sentence = carrier.replace("{SLOT}", phrase)

        # Tokenise by whitespace (we'll use the tokenizer's word_ids later)
        tokens = sentence.split()
        labels = ["O"] * len(tokens)

        # Find the sensitive phrase tokens and label them BIO
        phrase_tokens = phrase.split()
        for i in range(len(tokens) - len(phrase_tokens) + 1):
            window = tokens[i : i + len(phrase_tokens)]
            if [t.lower().strip(".,;:!?") for t in window] == [
                t.lower().strip(".,;:!?") for t in phrase_tokens
            ]:
                labels[i] = f"B-{cat}"
                for j in range(1, len(phrase_tokens)):
                    labels[i + j] = f"I-{cat}"
                break

        examples.append({"tokens": tokens, "labels": labels})

    # --- Negative examples (all O) ---
    for _ in range(n_negative):
        sent = rng.choice(_NEGATIVE_SENTENCES)
        tokens = sent.split()
        labels = ["O"] * len(tokens)
        examples.append({"tokens": tokens, "labels": labels})

    rng.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class _SensitiveNERDataset(Dataset):
    """Token-classification dataset that aligns word-level BIO labels to
    sub-word tokens produced by a HF tokenizer."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 128,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        tokens = ex["tokens"]
        word_labels = [self.label2id.get(l, 0) for l in ex["labels"]]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Align labels to sub-word tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned: List[int] = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)  # special tokens → ignore
            elif wid != prev_word_id:
                aligned.append(word_labels[wid] if wid < len(word_labels) else 0)
            else:
                # continuation sub-word → same I- label or ignore
                lbl = word_labels[wid] if wid < len(word_labels) else 0
                # If the word label is B-*, continuation sub-words become I-*
                lbl_name = ID2LABEL.get(lbl, "O")
                if lbl_name.startswith("B-"):
                    cat = lbl_name[2:]
                    lbl = self.label2id.get(f"I-{cat}", lbl)
                aligned.append(lbl)
            prev_word_id = wid

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sensitive_model(
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 3e-5,
    n_positive: int = 2000,
    n_negative: int = 1000,
    seed: int = 42,
) -> Path:
    """
    Fine-tune distilbert-base-multilingual-cased for sensitive-word NER.

    All data is synthetic — generated from ``ARTICLE9_PATTERNS`` seeds +
    luxury-domain carrier sentences.  No real client data is used.

    Returns the path to the saved model directory.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
    )

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Generating synthetic training data …")
    examples = _generate_training_examples(n_positive, n_negative, seed)
    split = int(len(examples) * 0.9)
    train_ex, val_ex = examples[:split], examples[split:]

    logger.info(f"Train: {len(train_ex)}  Val: {len(val_ex)}  Labels: {NUM_LABELS}")

    # Load tokenizer + base model
    base_model = "distilbert-base-multilingual-cased"
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")
    model.to(device)

    # Datasets
    train_ds = _SensitiveNERDataset(train_ex, tokenizer, LABEL2ID)
    val_ds = _SensitiveNERDataset(val_ex, tokenizer, LABEL2ID)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out.loss.item()
        avg_train = total_loss / len(train_dl)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += out.loss.item()
                preds = out.logits.argmax(dim=-1)
                mask = batch["labels"] != -100
                correct += (preds.cpu()[mask] == batch["labels"][mask]).sum().item()
                total += mask.sum().item()
        avg_val = val_loss / len(val_dl)
        acc = correct / total if total else 0

        logger.info(
            f"Epoch {epoch}/{epochs}  train_loss={avg_train:.4f}  "
            f"val_loss={avg_val:.4f}  val_acc={acc:.4f}"
        )
        print(
            f"  Epoch {epoch}/{epochs}  train_loss={avg_train:.4f}  "
            f"val_loss={avg_val:.4f}  val_acc={acc:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val

    # Save
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(_MODEL_DIR)
    tokenizer.save_pretrained(_MODEL_DIR)
    with open(_LABEL_MAP_FILE, "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f, indent=2)

    logger.info(f"Model saved to {_MODEL_DIR}")
    print(f"\n✅ Sensitive-NER model saved to {_MODEL_DIR}")
    return _MODEL_DIR


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class SensitiveWordDetector:
    """
    Load the fine-tuned NER model and predict sensitive tokens in text.

    Falls back gracefully if the model hasn't been trained yet.
    """

    def __init__(self, model_dir: Optional[Path] = None, threshold: float = 0.70):
        self.model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._device = None
        self._available = False

        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            try:
                self._load()
                self._available = True
            except Exception as e:
                logger.warning(f"Could not load sensitive-NER model: {e}")
        else:
            logger.info(
                "Sensitive-NER model not found — running without ML safety net. "
                "Train with: python -m server.privacy.sensitive_model train"
            )

    @property
    def available(self) -> bool:
        return self._available

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self._model.eval()

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)
        logger.info(f"Sensitive-NER model loaded from {self.model_dir} ({self._device})")

    def predict(self, text: str) -> List[Dict]:
        """
        Detect sensitive tokens in *text*.

        Returns a list of dicts::

            [
                {"token": "épilepsie", "label": "HEALTH_DATA",
                 "score": 0.97, "start": 23, "end": 33},
                ...
            ]
        """
        if not self._available:
            return []

        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = encoding.pop("offset_mapping").squeeze(0).tolist()

        with torch.no_grad():
            out = self._model(
                input_ids=encoding["input_ids"].to(self._device),
                attention_mask=encoding["attention_mask"].to(self._device),
            )
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).cpu()
        pred_ids = probs.argmax(dim=-1).tolist()
        max_scores = probs.max(dim=-1).values.tolist()

        results: List[Dict] = []
        for i, (pid, score) in enumerate(zip(pred_ids, max_scores)):
            lbl = ID2LABEL.get(pid, "O")
            if lbl == "O" or score < self.threshold:
                continue
            start, end = offsets[i]
            if start == end:
                continue  # special token

            # Strip B-/I- prefix to get category
            category = lbl.split("-", 1)[1] if "-" in lbl else lbl
            token_text = text[start:end]

            results.append({
                "token": token_text,
                "label": category,
                "score": round(score, 4),
                "start": start,
                "end": end,
            })

        # Merge consecutive tokens that belong to the same entity
        merged = self._merge_entities(results, text)
        return merged

    @staticmethod
    def _merge_entities(tokens: List[Dict], text: str) -> List[Dict]:
        """Merge consecutive sub-word / word tokens into single entities."""
        if not tokens:
            return []

        merged: List[Dict] = []
        current = dict(tokens[0])

        for tok in tokens[1:]:
            # Same category and adjacent (allowing for whitespace / sub-word gaps)
            if tok["label"] == current["label"] and tok["start"] - current["end"] <= 2:
                current["end"] = tok["end"]
                current["token"] = text[current["start"] : current["end"]]
                current["score"] = max(current["score"], tok["score"])
            else:
                merged.append(current)
                current = dict(tok)
        merged.append(current)
        return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for training / testing the model."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="RGPD Sensitive-Word NER model (train / test)"
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- train ---
    p_train = sub.add_parser("train", help="Train the sensitive-word NER model")
    p_train.add_argument("--epochs", type=int, default=6)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=3e-5)
    p_train.add_argument("--n-positive", type=int, default=2000)
    p_train.add_argument("--n-negative", type=int, default=1000)

    # --- test ---
    p_test = sub.add_parser("test", help="Run the model on sample texts")
    p_test.add_argument("--text", type=str, default=None, help="Text to classify")

    args = parser.parse_args()

    if args.cmd == "train":
        train_sensitive_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_positive=args.n_positive,
            n_negative=args.n_negative,
        )

    elif args.cmd == "test":
        det = SensitiveWordDetector()
        if not det.available:
            print("❌ Model not trained yet.  Run:  python -m server.privacy.sensitive_model train")
            return

        test_texts = [
            args.text
        ] if args.text else [
            "Le client souffre d'épilepsie sévère et cherche un sac hypoallergénique.",
            "She is muslim and wants halal-certified leather goods.",
            "Discussion about tissus ethniques et motifs traditionnels.",
            "He mentioned his criminal record during the consultation.",
            "La cliente est enceinte et cherche un cadeau pour son mari.",
            "Looking for a scarf in cashmere with geometric print.",
            "Budget 3500€ pour un sac Capucines en cuir noir.",
        ]

        print("\n🔍 Sensitive-Word NER Predictions\n")
        for txt in test_texts:
            hits = det.predict(txt)
            print(f"  Text: {txt}")
            if hits:
                for h in hits:
                    print(f"    ⚠️  [{h['label']}] \"{h['token']}\"  (score={h['score']:.2f}, pos={h['start']}:{h['end']})")
            else:
                print("    ✅ No sensitive words detected")
            print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
