#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_DIR="trained_adapters/best"
ONTOLOGY_PATH="ontologies/ExpertAssessmentSchema_UO_RE.rdf"
MISTAKE_TYPES_PATH="../prompts/modeling_mistakes.json"
REASONER="hermit"
MAX_LENGTH=2048

# ── Input sample ──────────────────────────────────────────────────────────────
SAMPLE_ID="sample_001"

I="Every Outfit suitable for an Event has to have exactly one bag."
OA=":EventOutfit rdf:type owl:Class ;
  rdfs:subClassOf :Outfit .

:suitableForEvent rdf:type owl:ObjectProperty ;
  rdfs:domain :EventOutfit ;
  rdfs:range :ClothingItem ,
  [ rdf:type owl:Restriction ;
  owl:onProperty :suitableForEvent ;
  owl:minQualifiedCardinality ""3""^^xsd:nonNegativeInteger ;
  owl:onClass :Jewelry
  ] ,
  [ rdf:type owl:Restriction ;
  owl:onProperty :suitableForEvent ;
  owl:qualifiedCardinality ""1""^^xsd:nonNegativeInteger ;
  owl:onClass :Bag
  ] ,
  [ rdf:type owl:Restriction ;
  owl:onProperty :suitableForEvent ;
  owl:qualifiedCardinality ""1""^^xsd:nonNegativeInteger ;
  owl:onClass :HighHeels
  ] ;
  rdfs:comment ""Clothing items suitable for events."" .

:Bag rdf:type owl:Class ;
  rdfs:subClassOf :Accessory ;
  rdfs:comment ""A bag is an accessory."" ."
M_I="opr_mistake_type_2"
GENERATION="The axiom incorrectly places cardinality restrictions in the range of the object property \`suitableForEvent\`, which implies that any \`ClothingItem\` suitable for an event must itself be associated with exactly one \`Bag\`, rather than ensuring that each \`EventOutfit\` has exactly one \`Bag\`, potentially leading to unintended inferences about individual clothing items."

OUTPUT_JSON="prediction_outputs/scored_explanation_${SAMPLE_ID}.json"

# ── Sanity checks ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICT_SCRIPT="${SCRIPT_DIR}/predict.py"

if [[ ! -f "${PREDICT_SCRIPT}" ]]; then
    echo "[ERROR] predict.py not found at: ${PREDICT_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "[ERROR] Adapter directory not found: ${ADAPTER_DIR}" >&2
    exit 1
fi

if ! command -v java &>/dev/null; then
    echo "[WARNING] Java not found on PATH — the OWL reasoner will fail." >&2
fi

# ── Run ───────────────────────────────────────────────────────────────────────
python3 "${PREDICT_SCRIPT}" \
    --I                  "${I}" \
    --OA                 "${OA}" \
    --M_i                "${M_I}" \
    --generation         "${GENERATION}" \
    --adapter_dir        "${ADAPTER_DIR}" \
    --ontology_path      "${ONTOLOGY_PATH}" \
    --mistake_types_path "${MISTAKE_TYPES_PATH}" \
    --sample_id          "${SAMPLE_ID}" \
    --max_length         "${MAX_LENGTH}" \
    --reasoner           "${REASONER}" \
    --output_json        "${OUTPUT_JSON}"