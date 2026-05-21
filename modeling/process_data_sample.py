import json
import uuid
from pathlib import Path

from owlready2 import World, Restriction, And, VALUE, sync_reasoner_hermit

# Paths
ONTOLOGY_PATH = Path("ontologies/ExpertAssessmentSchema_UO_RE.rdf")
MISTAKE_TYPES_PATH = Path("../prompts/modeling_mistakes.json")
OUTPUT_RDF_PATH = Path("processed_data/output_sample.rdf")
OUTPUT_JSON_PATH = Path("processed_data/output_sample.json")


def load_mistake_types(path):
    """
        Load the dict of mistake types
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_ind_name(prefix):
    """
        Make a safe individual name (process spaces and special chars to end up with safe URIs)
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def get_comment(ind):
    """
        Get the rdfs:comment string from an individual (first value or None)
    """
    comments = ind.comment
    return comments[0] if comments else None


def get_unintended_outcomes(mistake_subcls, hasUO_prop):
    """
        Extract UO dicts from the OPRMistake subclass axioms.
    """
    uo_list = []
    _collect_uo(mistake_subcls.is_a, hasUO_prop, uo_list)
    return uo_list


def _collect_uo(nodes, hasUO_prop, acc):
    for node in nodes:
        if isinstance(node, And):
            _collect_uo(node.Classes, hasUO_prop, acc)
        elif isinstance(node, Restriction):
            if node.property == hasUO_prop and node.type == VALUE:
                ind = node.value
                acc.append({"name": ind.name, "comment": get_comment(ind)})


def process_sample(input_data, ontology_path=ONTOLOGY_PATH, mistake_types_path=MISTAKE_TYPES_PATH, output_rdf=False, output_rdf_path=OUTPUT_RDF_PATH, output_json=False, output_json_path=OUTPUT_JSON_PATH, verbose=True):
    """
        Loads the ExpertAssessmentSchema ontology, creates individuals for a given data sample,
        runs the HermiT reasoner, saves the result into an RDF file, and returns an enriched
        JSON dict with the relevant elements (RE) and unintended outcomes (UO) for the sample.

        Expected data sample structure:
            {
                "OA"         : original axiom (turtle code) written by the student,
                "I"          : intended meaning sentence,
                "M_i"        : mistake type key matching opr_mistake_types.json,
                "generation" : generated explanation text
            }
    """
    # 1. Load supporting data
    mistake_types = load_mistake_types(mistake_types_path)

    mistake_key = input_data["M_i"]
    if mistake_key not in mistake_types:
        raise ValueError(
            f"Unknown mistake type '{mistake_key}'. "
            f"Valid keys: {list(mistake_types.keys())}"
        )
    mistake_entry         = mistake_types[mistake_key]
    ontology_concept_name = mistake_entry["ontology_concept"]
    mistake_description   = mistake_entry["description"]

    # 2. Load ontology in a new world
    world = World()
    onto = world.get_ontology(ontology_path.resolve().as_uri()).load()
    eas = onto.get_namespace("http://www.semanticweb.org/user/ontologies/2026/2/expertassessmentschema#")

    # 3. Retrieve required classes from the ontology
    Explanation_cls = eas.Explanation
    OPRMistake_cls = eas.OPRMistake
    InferenceCapability_cls = eas.InferenceCapability
    MistakeCorrectionCompetence_cls = eas.MistakeCorrectionCompetence
    ModelingComprehension_cls = eas.ModelingComprehension
    AppropriateVocabularyUsage_cls = eas.AppropriateVocabularyUsage

    # Get the specific OPRMistake subclass for this sample from the M_i in the data sample
    MistakeSubCls = getattr(eas, ontology_concept_name)
    if MistakeSubCls is None:
        raise ValueError(f"Class '{ontology_concept_name}' not found in the ontology.")

    # Object / data properties
    hasText_prop = eas.hasText
    explainsMistake_prop = eas.explainsMistake
    isPartOf_prop = eas.isPartOf
    hasRE_prop = eas.hasRelevantElement
    hasUO_prop = eas.hasUnintendedOutcome

    # 4. Create individuals within the ontology
    with onto:

        # 4a. Explanation
        explanation_ind = Explanation_cls(make_ind_name("explanation"))
        hasText_prop[explanation_ind] = [input_data["generation"]]

        # 4b. OPRMistake
        mistake_ind = MistakeSubCls(make_ind_name("mistake"))
        if OPRMistake_cls not in mistake_ind.is_a:
            mistake_ind.is_a.append(OPRMistake_cls)

        # 4c. Four QualityDimension individuals (sub-subclass classification is left for a later module)
        ic_ind  = InferenceCapability_cls(make_ind_name("inference_capability"))
        mcc_ind = MistakeCorrectionCompetence_cls(make_ind_name("mistake_correction_competence"))
        mc_ind  = ModelingComprehension_cls(make_ind_name("modeling_comprehension"))
        avu_ind = AppropriateVocabularyUsage_cls(make_ind_name("appropriate_vocabulary_usage"))

        # 4d. Object properties assertions
        explainsMistake_prop[explanation_ind] = [mistake_ind]
        if isPartOf_prop is not None:
            for dim_ind in [ic_ind, mcc_ind, mc_ind, avu_ind]:
                isPartOf_prop[dim_ind] = [explanation_ind]

    # 5. Run HermiT reasoner
    if verbose:
        print("Running HermiT reasoner …")
    with onto:
        debug = int(verbose)
        sync_reasoner_hermit(world, infer_property_values=True, debug=debug)
    if verbose:
        print("Reasoning complete.")

    # 6. Save the result into a RDF file (ontology + individuals + inferences)
    if output_rdf:
        onto.save(file=str(output_rdf_path), format="rdfxml")
        if verbose:
            print(f"RDF saved to: {output_rdf_path.resolve()}")

    # 7. Build and save enriched JSON output
    # REs and UOs are read from the inferred property values on mistake_ind
    relevant_elements = [
        {"name": re.name, "comment": get_comment(re)}
        for re in hasRE_prop[mistake_ind]
    ]
    unintended_outcomes = [
        {"name": uo.name, "comment": get_comment(uo)}
        for uo in hasUO_prop[mistake_ind]
    ]

    output = {
        "OA":                  input_data["OA"],
        "I":                   input_data["I"],
        "M_i":                 input_data["M_i"],
        "mistake_description": mistake_description,
        "generation":          input_data["generation"],
        "relevant_elements":   relevant_elements,
        "unintended_outcomes": unintended_outcomes,
    }

    if output_json:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"JSON saved to: {output_json_path.resolve()}")

    return output
