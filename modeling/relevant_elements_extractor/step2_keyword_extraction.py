"""
step2_keyword_extraction.py
============================
Implements the four-step keyword extraction pipeline described in the design:

  Step 1 — Preprocessing: lowercase, lemmatize, remove stopwords
  Step 2 — Exact/partial match against SKOS prefLabels and altLabels
  Step 3 — Fuzzy matching: edit-distance matching for near-misses (rapidfuzz)
  Step 4 — Concept resolution: map surface forms to skos:Concept URIs

For each mistake type corpus file, the output is a JSON file listing the resolved SKOS concept URIs with evidence (which surface form triggered each match).

Dependencies:
    pip install rdflib rapidfuzz spacy
    python -m spacy download en_core_web_sm

Usage:
    python3 step2_keyword_extraction.py \
        --vocabulary opr_relevant_elements.ttl \
        --corpus-dir corpus/ \
        --output-dir output/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Dependency imports 
try:
    from rdflib import Graph, Namespace, URIRef
    from rdflib.namespace import SKOS
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    print("[WARN] rdflib not found.", file=sys.stderr)

try:
    from rapidfuzz import fuzz, process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    import difflib
    print("[WARN] rapidfuzz not found. Falling back to difflib for fuzzy matching.", file=sys.stderr)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Constants
FUZZY_THRESHOLD = 88        # minimum similarity score (0–100) for fuzzy match
NGRAM_MAX       = 5         # maximum token n-gram size to consider
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "as", "into", "through", "and",
    "or", "not", "no", "nor", "but", "if", "then", "that", "this",
    "which", "who", "what", "when", "where", "how", "its", "it", "their",
    "they", "he", "she", "we", "you", "i", "thus", "also", "such", "so",
    "even", "both", "each", "any", "all", "more", "most", "only", "other",
    "rather", "than", "there", "here", "very", "often", "always", "never"}

# Vocabulary loader
class SKOSVocabulary:
    """
        Loads the SKOS .ttl vocabulary and exposes:
        - concept_labels: uri → {"prefLabel": str, "altLabels": [str]}
        - label_to_uri:   normalised_label → [uri, ...]   (multi-map)
        - related:        uri → [uri, ...]                 (implication table)
        - all_labels:     flat sorted list of all normalised labels (for fuzzy index)
    """

    def __init__(self, ttl_path: Path):
        self.ttl_path = ttl_path
        self.concept_labels: dict[str, dict] = {}
        self.label_to_uri:   dict[str, list[str]] = defaultdict(list)
        self.related:        dict[str, list[str]] = defaultdict(list)
        self._load()
        self.all_labels = sorted(self.label_to_uri.keys())

    @staticmethod
    def _normalise(label: str) -> str:
        return label.lower().strip()

    def _load(self):
        if RDFLIB_AVAILABLE:
            self._load_rdflib()
        else:
            print("[ERROR] rdflib is required to load the SKOS concepts.")
            exit(1)

    def _load_rdflib(self):
        g = Graph()
        g.parse(str(self.ttl_path), format="turtle")
        SKOS_NS = Namespace("http://www.w3.org/2004/02/skos/core#")

        for s, p, o in g.triples((None, SKOS_NS.prefLabel, None)):
            uri = str(s)
            label = self._normalise(str(o))
            if uri not in self.concept_labels:
                self.concept_labels[uri] = {"prefLabel": label, "altLabels": []}
            else:
                self.concept_labels[uri]["prefLabel"] = label
            self.label_to_uri[label].append(uri)

        for s, p, o in g.triples((None, SKOS_NS.altLabel, None)):
            uri = str(s)
            label = self._normalise(str(o))
            if uri not in self.concept_labels:
                self.concept_labels[uri] = {"prefLabel": None, "altLabels": [label]}
            else:
                self.concept_labels[uri]["altLabels"].append(label)
            self.label_to_uri[label].append(uri)

        for s, p, o in g.triples((None, SKOS_NS.related, None)):
            self.related[str(s)].append(str(o))
            self.related[str(o)].append(str(s))

    def get_concept_name(self, uri: str) -> str:
        """
            Human-readable name for a concept URI (prefLabel or local name).
        """
        info = self.concept_labels.get(uri, {})
        if info.get("prefLabel"):
            return info["prefLabel"]
        return uri.split("#")[-1].split("/")[-1]


# Step 1 — Preprocessing
def preprocess(text, nlp=None):
    """
        Returns (lowercased_text, lemmatized_tokens).
        Uses spaCy lemmatizer if available, otherwise basic whitespace tokenisation.
    """
    text_lower = text.lower()
    text_clean = re.sub(r"[^\w\s\-]", " ", text_lower) # Remove punctuation except hyphens (which appear in OWL term names)

    if nlp is not None and SPACY_AVAILABLE:
        doc = nlp(text_clean)
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha and token.lemma_ not in STOP_WORDS
        ]
    else:
        raw_tokens = text_clean.split()
        tokens = [t for t in raw_tokens if t not in STOP_WORDS and len(t) > 1]

    return tokens


def make_ngrams(tokens, max_n=NGRAM_MAX):
    """Generate all n-grams (n=1..max_n) from a token list."""
    ngrams = []
    for n in range(1, min(max_n + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i:i+n]))
    return ngrams


# Step 2 — Exact/partial match
def exact_match(ngrams, vocab):
    """
        Perform an exact lookup of n-grams against all SKOS labels.
        Returns {surface_form → [concept_uri, ...]}
    """
    matches = {}
    for ng in ngrams:
        if ng in vocab.label_to_uri:
            matches[ng] = vocab.label_to_uri[ng]
    return matches

# Step 3 — Fuzzy matching
def fuzzy_match(ngrams, vocab, threshold=FUZZY_THRESHOLD):
    """
        Returns {surface_form → [concept_uri, ...]}
        Only applies to n-grams that were NOT already matched exactly.
    """
    matches = {}
    if not vocab.all_labels:
        return matches

    for ng in ngrams:
        if RAPIDFUZZ_AVAILABLE:
            results = rf_process.extract(
                ng, vocab.all_labels,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
                limit=3,
            )
            for matched_label, score, _ in results:
                if matched_label in vocab.label_to_uri:
                    matches.setdefault(f"{ng}~{matched_label}", []).extend(
                        vocab.label_to_uri[matched_label]
                    )
        else:
            # difflib fallback
            close = difflib.get_close_matches(
                ng, vocab.all_labels, n=3, cutoff=threshold / 100.0
            )
            for matched_label in close:
                if matched_label in vocab.label_to_uri:
                    matches.setdefault(f"{ng}~{matched_label}", []).extend(
                        vocab.label_to_uri[matched_label]
                    )
    return matches

# Step 4 — Concept resolution
def resolve_and_expand(raw_matches, vocab):
    """
        Returns {concept_uri → {"concept_name": str, "triggered_by": [surface_form, ...],
                        "implied_by": [source_uri, ...]}}
    """
    resolved: dict[str, dict] = {}

    for surface, uris in raw_matches.items():
        for uri in set(uris):
            if uri not in resolved:
                resolved[uri] = {
                    "concept_name": vocab.get_concept_name(uri),
                    "triggered_by": [],
                    "implied_by": [],
                }
            resolved[uri]["triggered_by"].append(surface)

    # Deduplicate triggered_by lists
    for uri in resolved:
        resolved[uri]["triggered_by"] = sorted(set(resolved[uri]["triggered_by"]))

    return resolved


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def process_document(text, vocab, nlp=None):
    """
        Run the full four-step pipeline on a single text string.
    """

    # Step 1 — Preprocessing
    tokens = preprocess(text, nlp)

    # Build n-grams from lemmatized tokens
    ngrams = make_ngrams(tokens)

    # Step 2 — Exact match
    exact = exact_match(ngrams, vocab)

    # Step 3 — Fuzzy match (on ngrams NOT already exactly matched)
    already_matched_surfaces = set(exact.keys())
    remaining_ngrams = [ng for ng in ngrams if ng not in already_matched_surfaces]
    fuzzy = fuzzy_match(remaining_ngrams, vocab)

    # Merge exact + fuzzy
    all_raw = {**exact, **fuzzy}

    # Step 4 — Concept resolution + implication expansion
    resolved = resolve_and_expand(all_raw, vocab)

    return {
        "n_tokens":        len(tokens),
        "n_ngrams":        len(ngrams),
        "n_exact_matches": len(exact),
        "n_fuzzy_matches": len(fuzzy),
        "concepts":        resolved,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract SKOS concept matches from OPR mistake type corpus.")
    parser.add_argument("--vocabulary", type=Path, default=Path("opr_relevant_elements.ttl"), help="Path to the SKOS vocabulary TTL file.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"), help="Directory containing corpus .txt files and manifest.json.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory where extraction results will be written.")
    args = parser.parse_args()

    # Validate paths
    if not args.vocabulary.exists():
        print(f"[ERROR] Vocabulary file not found: {args.vocabulary}", file=sys.stderr)
        sys.exit(1)

    manifest_path = args.corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}. Run step1 first.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    print(f"[INFO] Loading SKOS vocabulary: {args.vocabulary}")
    vocab = SKOSVocabulary(args.vocabulary)
    print(f"[INFO] Vocabulary loaded: {len(vocab.concept_labels)} concepts, "
          f"{len(vocab.label_to_uri)} distinct labels")

    # Load spaCy if available
    nlp = None
    if SPACY_AVAILABLE:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("[INFO] spaCy model loaded (en_core_web_sm).")
        except OSError:
            print("[WARN] spaCy model en_core_web_sm not found. "
                  "Run: python -m spacy download en_core_web_sm", file=sys.stderr)

    # Load manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_results = {}

    for entry in manifest:
        corpus_file = Path(entry["corpus_file"])
        short_id    = entry["short_id"]
        local_name  = entry["local_name"]

        if not corpus_file.exists():
            print(f"[WARN] Corpus file missing: {corpus_file}")
            continue

        text = corpus_file.read_text(encoding="utf-8")
        print(f"\n[INFO] Processing: {short_id}")

        result = process_document(text, vocab, nlp)

        for _, info in sorted(result["concepts"].items(),
                                key=lambda x: x[1]["concept_name"]):
            tag = "(implied)" if not info["triggered_by"] else ""
            print(f"    • {info['concept_name']} {tag}")
            if info["triggered_by"]:
                triggers = ", ".join(info["triggered_by"][:3])
                if len(info["triggered_by"]) > 3:
                    triggers += f" (+{len(info['triggered_by'])-3} more)"
                print(f"      triggered by: {triggers}")

        all_results[short_id] = {
            "local_name": local_name,
            "uri":        entry["uri"],
            **result,
        }

        # Write per-type result file
        out_file = args.output_dir / f"{short_id}_concepts.json"
        out_file.write_text(
            json.dumps(all_results[short_id], indent=2), encoding="utf-8"
        )

    # Write combined results summary
    summary_path = args.output_dir / "extraction_results.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[INFO] Step 2 complete. Results written to {args.output_dir}/")
    print(f"[INFO] Combined summary: {summary_path}")


if __name__ == "__main__":
    main()
