#!/usr/bin/env bash
# =============================================================================
# extract_and_populate_relevant_elements.sh
# =============================================================================
# Orchestrates the three-step OPR keyword extraction and ontology population
# pipeline:
#
#   Step 1 — Extract rdfs:comment corpus from ontology  (step1_extract_corpus.py)
#   Step 2 — Keyword extraction via SKOS vocabulary     (step2_keyword_extraction.py)
#   Step 3 — Populate ontology with relevant elements   (step3_populate_ontology.py)
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh [OPTIONS]
#
# Options:
#   --ontology  PATH   Path to the RDF ontology (default: ExpertAssessmentSchema.rdf)
#   --vocab     PATH   Path to SKOS vocabulary TTL (default: owl2_opr_vocabulary.ttl)
#   --corpus    DIR    Corpus output directory (default: corpus/)
#   --output    DIR    Extraction results directory (default: output/)
#   --enriched  PATH   Enriched ontology output path
#                      (default: output/ExpertAssessmentSchema_enriched.rdf)
#   --install-deps     Run pip install for required libraries before pipeline
#   --help             Show this message
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONTOLOGY="$(realpath "${SCRIPT_DIR}/../ontologies/ExpertAssessmentSchema_UO.rdf")"
VOCAB="${SCRIPT_DIR}/ontologies/opr_relevant_elements.ttl"
CORPUS_DIR="${SCRIPT_DIR}/corpus"
OUTPUT_DIR="${SCRIPT_DIR}/enriched_onto_outputs"
ENRICHED="${OUTPUT_DIR}/ExpertAssessmentSchema_enriched.rdf"
INSTALL_DEPS=false

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()   { echo -e "${CYAN}[$(date +%H:%M:%S)]${RESET} $*"; }
ok()    { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
error() { echo -e "${RED}[✗] $*${RESET}"; exit 1; }

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ontology)    ONTOLOGY="$2";  shift 2 ;;
        --vocab)       VOCAB="$2";     shift 2 ;;
        --corpus)      CORPUS_DIR="$2"; shift 2 ;;
        --output)      OUTPUT_DIR="$2"; shift 2 ;;
        --enriched)    ENRICHED="$2";  shift 2 ;;
        --install-deps) INSTALL_DEPS=true; shift ;;
        --help|-h)     usage ;;
        *) warn "Unknown option: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "${CORPUS_DIR}" "${OUTPUT_DIR}"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   OPR Keyword Extraction & Ontology Population Pipeline      ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
log "Pipeline started"
log "Ontology  : ${ONTOLOGY}"
log "Vocabulary: ${VOCAB}"
log "Corpus dir: ${CORPUS_DIR}"
log "Output dir: ${OUTPUT_DIR}"
log "Enriched  : ${ENRICHED}"
echo ""

# ---------------------------------------------------------------------------
# Python interpreter check
# ---------------------------------------------------------------------------
PYTHON=""
for py in python3 python; do
    if command -v "$py" &>/dev/null; then
        PYTHON="$py"
        break
    fi
done
[[ -z "$PYTHON" ]] && error "Python 3 not found. Please install Python 3.8+."
PY_VERSION="$("${PYTHON}" --version 2>&1)"
log "Python: ${PY_VERSION} (${PYTHON})"

# ---------------------------------------------------------------------------
# Optional dependency installation
# ---------------------------------------------------------------------------
if [[ "${INSTALL_DEPS}" == true ]]; then
    echo ""
    log "Installing Python dependencies..."
    "${PYTHON}" -m pip install rdflib rapidfuzz spacy --break-system-packages -q \
        && ok "rdflib, rapidfuzz, spacy installed." \
        || warn "Some packages failed to install — pipeline will use fallbacks."
    "${PYTHON}" -m spacy download en_core_web_sm -q \
        && ok "spaCy model en_core_web_sm downloaded." \
        || warn "spaCy model download failed — syntactic expansion will be skipped."
    echo ""
fi

# Report which optional libraries are available
echo ""
log "Checking optional library availability:"
for lib in rdflib rapidfuzz spacy; do
    if "${PYTHON}" -c "import ${lib}" 2>/dev/null; then
        ok "  ${lib} ✓"
    else
        warn "  ${lib} ✗ (fallback will be used)"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
[[ -f "${ONTOLOGY}" ]] || error "Ontology file not found: ${ONTOLOGY}"
[[ -f "${VOCAB}"    ]] || error "Vocabulary file not found: ${VOCAB}"

# ---------------------------------------------------------------------------
# Step 1 — Extract corpus
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}━━━ Step 1: Extract rdfs:comment corpus ━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
log "Running step1_extract_corpus.py..."

STEP1_START=$(date +%s)
"${PYTHON}" "${SCRIPT_DIR}/step1_rdfs_comment_extraction.py" \
    --ontology  "${ONTOLOGY}" \
    --output-dir "${CORPUS_DIR}" \

STEP1_EXIT="${PIPESTATUS[0]}"
STEP1_END=$(date +%s)

if [[ "${STEP1_EXIT}" -eq 0 ]]; then
    ok "Step 1 completed in $(( STEP1_END - STEP1_START ))s."
    N_CORPUS=$(find "${CORPUS_DIR}" -name "*.txt" | wc -l | tr -d ' ')
    log "  Corpus files created: ${N_CORPUS}"
else
    error "Step 1 failed with exit code ${STEP1_EXIT}."
fi

# ---------------------------------------------------------------------------
# Step 2 — Keyword extraction
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}━━━ Step 2: Keyword extraction (SKOS matching) ━━━━━━━━━━━━━━━━${RESET}"
log "Running step2_keyword_extraction.py..."

STEP2_START=$(date +%s)
"${PYTHON}" "${SCRIPT_DIR}/step2_keyword_extraction.py" \
    --vocabulary "${VOCAB}" \
    --corpus-dir "${CORPUS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \

STEP2_EXIT="${PIPESTATUS[0]}"
STEP2_END=$(date +%s)

if [[ "${STEP2_EXIT}" -eq 0 ]]; then
    ok "Step 2 completed in $(( STEP2_END - STEP2_START ))s."
    N_JSON=$(find "${OUTPUT_DIR}" -name "*_concepts.json" | wc -l | tr -d ' ')
    log "  Concept files created: ${N_JSON}"
else
    error "Step 2 failed with exit code ${STEP2_EXIT}."
fi

# ---------------------------------------------------------------------------
# Step 3 — Ontology population
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}━━━ Step 3: Ontology population ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
log "Running step3_populate_ontology.py..."

STEP3_START=$(date +%s)
"${PYTHON}" "${SCRIPT_DIR}/step3_ontology_population.py" \
    --ontology   "${ONTOLOGY}" \
    --vocabulary "${VOCAB}" \
    --results-dir "${OUTPUT_DIR}" \
    --output     "${ENRICHED}" \

STEP3_EXIT="${PIPESTATUS[0]}"
STEP3_END=$(date +%s)

if [[ "${STEP3_EXIT}" -eq 0 ]]; then
    ok "Step 3 completed in $(( STEP3_END - STEP3_START ))s."
else
    error "Step 3 failed with exit code ${STEP3_EXIT}."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$(( STEP3_END - STEP1_START ))
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Pipeline Complete                                          ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
ok "Total runtime: ${TOTAL}s"
echo ""
log "Outputs:"
log "  Corpus files  : ${CORPUS_DIR}/*.txt"
log "  Concept JSON  : ${OUTPUT_DIR}/*_concepts.json"
log "  Combined JSON : ${OUTPUT_DIR}/extraction_results.json"
log "  Enriched OWL  : ${ENRICHED}"
echo ""