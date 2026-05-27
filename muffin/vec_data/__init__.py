from .schema import (
    EvidenceStatus,
    ClaimType,
    VisualClaim,
    VecDPOSample,
    compute_response_evidence_score,
    load_vec_dpo_json,
    save_vec_dpo_json,
    validate_vec_dpo_sample,
    validate_vec_dpo_dataset,
)

from .evidence_scorer import (
    EvidenceScoringConfig,
    EvidenceScorer,
    score_vec_dpo_samples,
    compute_dataset_statistics,
)

from .claim_extractor import (
    ClaimExtractionConfig,
    VisualClaimExtractor,
    extract_claims_from_raw_item,
    extract_claims_from_response_dict,
)

from .evidence_verifier import (
    EvidenceVerificationConfig,
    VerificationResult,
    VisualEvidenceVerifier,
    verify_raw_item,
    verify_response_dict,
    compute_verification_stats,
)