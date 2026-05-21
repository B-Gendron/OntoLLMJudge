"""
step1_rdfs_comment_extraction.py
=======================
Extracts rdfs:comment text for each OPRMistakeType (I–VI + Other) from the ExpertAssessmentSchema RDF ontology and writes one .txt file per type into the corpus/ directory.

Usage:
    python3 step1_rdfs_comment_extraction.py --ontology <path/to/ontology.rdf> --output-dir <path/to/corpus/>
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import sys

# RDF/OWL namespaces
NS = {
    "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl":  "http://www.w3.org/2002/07/owl#",
}

ONTOLOGY_BASE = "http://www.semanticweb.org/user/ontologies/2026/2/expertassessmentschema#"

# Mapping mistake types to their short identifiers
MISTAKE_SHORT_IDS = {
    "OPRMistakeType1_DomainRangeInconsistency":  "type1_domain_range_inconsistency",
    "OPRMistakeType2_DomainRangeMisplacement":   "type2_domain_range_misplacement",
    "OPRMistakeType3_EquivalenceMisuse":         "type3_equivalence_misuse",
    "OPRMistakeType4_LogicalMisunderstanding":   "type4_logical_misunderstanding",
    "OPRMistakeType5_TrivialSatisfaction":       "type5_trivial_satisfaction",
    "OPRMistakeType6_WrongClass":                "type6_wrong_class",
    "OPRMistakeTypeOther":                       "type_other",
}


def parse_ontology(ontology_path):
    """
        Parse the RDF/XML ontology and return the root element.
    """
    try:
        tree = ET.parse(ontology_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] Could not parse ontology XML: {e}", file=sys.stderr)
        sys.exit(1)


def build_uri(local_name):
    return ONTOLOGY_BASE + local_name


def extract_comments_for_class(root, class_uri):
    """
        Return all rdfs:comment text values found on the owl:Class element
        whose rdf:about attribute equals class_uri.
        Also collects comments from any rdfs:subClassOf restriction sub-trees
        that carry a comment.
    """
    comments = []
    for cls_elem in root.iter(f"{{{NS['owl']}}}Class"):
        about = cls_elem.get(f"{{{NS['rdf']}}}about")
        if about != class_uri:
            continue

        for comment in cls_elem.findall(f"{{{NS['rdfs']}}}comment"):
            text = (comment.text or "").strip()
            if text:
                comments.append(text)

    return comments


def write_corpus_file(output_dir, short_id, class_comments):
    """
        Write a structured .txt corpus file for one mistake type.
    """
    filepath = output_dir / f"{short_id}.txt"

    lines = []
    if class_comments:
        for c in class_comments:
            lines.append(c)
            lines.append("")
    else:
        lines.append("[No class-level comment found]")
        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Extract rdfs:comment corpus from ExpertAssessmentSchema for each OPR mistake type.")
    parser.add_argument("--ontology", type=Path, default=Path("ExpertAssessmentSchema.rdf"), help="Path to the RDF ontology file (default: ExpertAssessmentSchema.rdf)")
    parser.add_argument("--output-dir", type=Path, default=Path("corpus"), help="Directory where corpus .txt files will be written (default: corpus/)")
    args = parser.parse_args()

    if not args.ontology.exists():
        print(f"[ERROR] Ontology file not found: {args.ontology}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Parsing ontology: {args.ontology}")
    root = parse_ontology(args.ontology)

    manifest = []  # track what was written, for the next step

    for local_name in MISTAKE_SHORT_IDS.keys():
        class_uri = build_uri(local_name)
        short_id  = MISTAKE_SHORT_IDS[local_name]

        class_comments = extract_comments_for_class(root, class_uri)

        filepath = write_corpus_file(args.output_dir, short_id, class_comments)

        n_class    = len(class_comments)
        print(f"  [{short_id}] class_comments={n_class} → {filepath}")

        manifest.append({
            "local_name": local_name,
            "short_id":   short_id,
            "uri":        class_uri,
            "corpus_file": str(filepath),
            "n_class_comments":   n_class,
        })

    # Write a JSON manifest that step 2 will use
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[INFO] Manifest written to: {manifest_path}")
    print(f"[INFO] Step 1 complete. {len(manifest)} corpus files written to {args.output_dir}/")


if __name__ == "__main__":
    main()
