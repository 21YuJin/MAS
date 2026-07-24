"""
[Step 7D] Registry-Based Feature Validation and Screening Preparation.

This package validates candidate_feature_registry.json (Step 7C) for
structural integrity, availability, redundancy, leakage risk, known
confounds, and deployment feasibility, and writes a STATIC screening plan
for Step 8. It never removes a feature, fits normal statistics, computes a
correlation, trains LightGAE, or scores/selects a Reduced Core Set -- every
one of those is explicitly out of scope until a later step.

Modules:
    registry_validator     -- duplicate names, missing keys, dangling/cyclic dependencies
    availability_validator  -- available_now_mock / requires_ollama_runtime /
                                requires_normal_statistics / planned_not_available,
                                cross-checked against feature_generation_manifest.json
    redundancy_validator     -- groups mathematically_dependent_on/potentially_redundant_with
                                 relationships; recommends (never removes) a representative
    leakage_validator          -- scans formula/source_fields/notes for forbidden or
                                   dataset-provenance terms
    confound_validator          -- surfaces registry-declared known_confound entries plus
                                    WARNING-only additional watch candidates
    deployment_validator          -- portable / ollama_specific / runtime_specific /
                                      offline_only / diagnostic_only classification
    screening_plan                 -- builds the 10-stage Step 8 screening plan (plan only)
    report                          -- orchestrates all of the above and writes the Step 7D
                                        report set under reports/travel_a2a/feature_pool/
"""
