"""
step3_ontology_population.py
===========================
Populates the EAS ontology by:
1. Mapping extracted concepts to 're_slug' individuals in the VOCAB_NS.
2. Creating a single intersection for each Mistake Type containing:
   - hasValue restrictions pointing to these individuals.
   - Cardinality restrictions.
"""

import argparse
import json
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

try:
    from rdflib import Graph, Namespace, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL, XSD
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

# Namespaces
ONTOLOGY_NS  = "http://www.semanticweb.org/user/ontologies/2026/2/expertassessmentschema#"
VOCAB_NS     = "http://www.semanticweb.org/user/vocabularies/owl2-opr#"

HAS_RELEVANT_ELEMENT    = ONTOLOGY_NS + "hasRelevantElement"
HAS_UNINTENDED_OUTCOME  = ONTOLOGY_NS + "hasUnintendedOutcome"
OPR_MISTAKE_CLASS       = ONTOLOGY_NS + "OPRMistake"
OPR_RELEVANT_ELEM_CLASS = ONTOLOGY_NS + "OPRModelingRelevantElement"
OPR_UO_CLASS            = ONTOLOGY_NS + "OPRMistakeUnintendedOutcome"
EXTRACTION_METHOD_PROP  = ONTOLOGY_NS + "extractionMethod"
EXTRACTION_DATE_PROP    = ONTOLOGY_NS + "extractionDate"


def get_re_uri(concept_key):
    """
        Map a concept (URI fragment or label) to the 're_slug' individual URI.
        Example: 'Existential Restriction' -> '...#re_existential_restriction'
    """
    name = concept_key.split("#")[-1].split("/")[-1]
    slug = name.lower().strip()
    slug = re.sub(r'[\s\-]+', '_', slug)

    if not slug.startswith("re_"):
        slug = f"re_{slug}"
    return URIRef(VOCAB_NS + slug)

def read_existing_uo_uris(g, class_uri):
    """
        Read the existing UnintendedOutcome individuals from the class definition.
    """
    uo_prop = URIRef(HAS_UNINTENDED_OUTCOME)
    uo_uris = set()
    
    def collect(node):
        on_prop = g.value(node, OWL.onProperty)
        has_val = g.value(node, OWL.hasValue)
        if on_prop == uo_prop and has_val:
            uo_uris.add(has_val)
        intersection = g.value(node, OWL.intersectionOf)
        if intersection:
            for item in _rdf_list_items(g, intersection):
                if isinstance(item, BNode): collect(item)

    for _, _, super_expr in g.triples((class_uri, RDFS.subClassOf, None)):
        if isinstance(super_expr, BNode): collect(super_expr)
    return sorted(list(uo_uris))


def remove_old_intersections(g, class_uri):
    """
        Removes previous complex subClassOf intersections to prevent duplication.
    """
    to_remove = []
    for _, _, super_expr in g.triples((class_uri, RDFS.subClassOf, None)):
        if isinstance(super_expr, BNode) and g.value(super_expr, OWL.intersectionOf):
            to_remove.append(super_expr)
    for node in to_remove:
        g.remove((class_uri, RDFS.subClassOf, node))


def _rdf_list_items(g, list_node):
    curr = list_node
    while curr and curr != RDF.nil:
        yield g.value(curr, RDF.first)
        curr = g.value(curr, RDF.rest)


def build_rdf_list(g, items):
    if not items: return RDF.nil
    head = BNode()
    g.add((head, RDF.first, items[0]))
    g.add((head, RDF.rest, build_rdf_list(g, items[1:])))
    return head


def make_restriction(g, prop, val=None, card=None, on_class=None):
    r = BNode()
    g.add((r, RDF.type, OWL.Restriction))
    g.add((r, OWL.onProperty, prop))
    if val: g.add((r, OWL.hasValue, val))
    if card is not None:
        g.add((r, OWL.qualifiedCardinality, Literal(card, datatype=XSD.nonNegativeInteger)))
        g.add((r, OWL.onClass, on_class))
    return r


def populate(ontology_path, results, output_path):
    g = Graph()
    g.parse(str(ontology_path), format="xml")
    g.bind("opr", Namespace(ONTOLOGY_NS))
    g.bind("voc", Namespace(VOCAB_NS))

    rel_prop = URIRef(HAS_RELEVANT_ELEMENT)
    uo_prop  = URIRef(HAS_UNINTENDED_OUTCOME)

    # Process Mistake Types
    for short_id, data in results.items():
        class_uri = URIRef(data["uri"])
        concepts = data.get("concepts", {})
        
        # 1. Fetch existing UOs and clear existing complex definitions
        uo_uris = read_existing_uo_uris(g, class_uri)
        remove_old_intersections(g, class_uri)

        # 2. Map REs to individuals and declare them
        re_uris = []
        for key in concepts.keys():
            re_uri = get_re_uri(key)
            re_uris.append(re_uri)
            # Declare the individual in the ontology
            g.add((re_uri, RDF.type, OWL.NamedIndividual))
            g.add((re_uri, RDF.type, URIRef(OPR_RELEVANT_ELEM_CLASS)))

        re_uris = sorted(list(set(re_uris)))

        # 3. Build the new single intersection
        elements = [URIRef(OPR_MISTAKE_CLASS)]
        
        for re_uri in re_uris:
            elements.append(make_restriction(g, rel_prop, val=re_uri))
        
        for uo_uri in uo_uris:
            elements.append(make_restriction(g, uo_prop, val=uo_uri))

        if re_uris:
            elements.append(make_restriction(g, rel_prop, card=len(re_uris), on_class=URIRef(OPR_RELEVANT_ELEM_CLASS)))
        
        if uo_uris:
            elements.append(make_restriction(g, uo_prop, card=len(uo_uris), on_class=URIRef(OPR_UO_CLASS)))

        # Link to class via subClassOf
        anon_intersection = BNode()
        g.add((anon_intersection, RDF.type, OWL.Class))
        g.add((anon_intersection, OWL.intersectionOf, build_rdf_list(g, elements)))
        g.add((class_uri, RDFS.subClassOf, anon_intersection))

        print(f"[INFO] Processed {short_id}: {len(re_uris)} REs, {len(uo_uris)} UOs.")

    g.serialize(destination=str(output_path), format="pretty-xml")
    print(f"[SUCCESS] Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True) # RESTORED
    parser.add_argument("--results-dir", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, default=Path("output/ExpertAssessmentSchema_UO_RE.rdf"))
    args = parser.parse_args()

    if not RDFLIB_AVAILABLE:
        sys.exit("[ERROR] rdflib not installed.")

    results_file = args.results_dir / "extraction_results.json"
    if not results_file.exists():
        sys.exit(f"[ERROR] {results_file} missing.")
    
    results = json.loads(results_file.read_text(encoding="utf-8"))
    populate(args.ontology, results, args.output)

if __name__ == "__main__":
    main()