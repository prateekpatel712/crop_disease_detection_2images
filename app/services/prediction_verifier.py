from __future__ import annotations

import re
from typing import Any

from app.models.runtime import VerificationCheck, VerificationResult


class PredictionVerificationService:
    _placeholder_tokens = {
        "",
        "unknown",
        "model_not_uploaded",
        "runtime_not_connected",
        "not_available",
    }

    def verify(self, state: dict[str, Any]) -> VerificationResult:
        checks: list[VerificationCheck] = []
        issues: list[str] = []
        score = 0.0
        matched_evidence_count = 0

        image_ok = bool(state.get("image_ok", False))
        quality_status = state.get("quality_gate_status", "unknown")
        is_leaf = state.get("is_leaf")
        is_healthy = state.get("is_healthy")
        crop_prediction = state.get("crop_prediction", {})
        disease_prediction = state.get("disease_prediction", {})
        disease_route = state.get("disease_route", {})
        rag = state.get("rag", {})

        if not image_ok:
            return VerificationResult(
                status="blocked_input",
                verified=False,
                score=0.0,
                summary="Verification stopped because the uploaded image failed basic validation.",
                recommended_action="Upload a supported crop leaf image and try again.",
                issues=["Image validation failed."],
                checks=[
                    VerificationCheck(
                        name="image_validation",
                        status="failed",
                        details="The input did not pass basic validation checks.",
                        passed=False,
                    )
                ],
            )

        if quality_status == "stop":
            return VerificationResult(
                status="blocked_quality",
                verified=False,
                score=0.0,
                summary="Verification stopped because the image quality is too weak for reliable analysis.",
                recommended_action="Upload a brighter, sharper, higher-resolution image focused on a single leaf.",
                issues=["Image quality gate returned stop."],
                checks=[
                    VerificationCheck(
                        name="quality_gate",
                        status="failed",
                        details="The quality screening blocked the request.",
                        passed=False,
                    )
                ],
            )

        if is_leaf is False:
            return VerificationResult(
                status="blocked_non_leaf",
                verified=False,
                score=0.0,
                summary="Verification stopped because the system could not confirm a leaf in the image.",
                recommended_action="Upload a tighter crop centered on one crop leaf.",
                issues=["Leaf detection failed."],
                checks=[
                    VerificationCheck(
                        name="leaf_detection",
                        status="failed",
                        details="The image does not look like a crop leaf input.",
                        passed=False,
                    )
                ],
            )

        quality_check = VerificationCheck(
            name="quality_gate",
            status="passed" if quality_status == "proceed" else "caution",
            details=f"Quality status is `{quality_status}`.",
            passed=quality_status == "proceed",
        )
        checks.append(quality_check)
        score += 0.10 if quality_status == "proceed" else 0.05

        leaf_check = VerificationCheck(
            name="leaf_detection",
            status="passed" if is_leaf is True else "caution",
            details="Leaf screening completed." if is_leaf is True else "Leaf screening is uncertain.",
            passed=is_leaf is True if is_leaf is not None else None,
        )
        checks.append(leaf_check)
        if is_leaf is True:
            score += 0.10

        if is_healthy is True and not bool(disease_prediction.get("model_ready", False)):
            checks.append(
                VerificationCheck(
                    name="healthy_leaf_gate",
                    status="passed",
                    details="The healthy-leaf screen skipped disease analysis for this image.",
                    passed=True,
                )
            )
            return VerificationResult(
                status="healthy_screen_skip",
                verified=True,
                score=round(min(score + 0.15, 1.0), 4),
                summary="The pipeline confidently routed this image through the healthy-leaf branch, so disease verification was not required.",
                recommended_action="If the leaf still looks suspicious to you, upload another image or continue after disease-model integration.",
                issues=[],
                matched_evidence_count=0,
                checks=checks,
            )

        crop_label = self._normalize_label(crop_prediction.get("label"))
        crop_ready = bool(crop_prediction.get("model_ready", False))
        crop_supported = self._label_in_supported_labels(
            crop_prediction.get("label"),
            crop_prediction.get("supported_labels", []),
        )
        crop_status = "pending"
        crop_details = "Crop model runtime is not connected yet."
        crop_passed: bool | None = None
        if crop_ready and crop_label and crop_label not in self._placeholder_tokens:
            crop_status = "passed"
            crop_details = "Crop prediction produced a concrete label."
            crop_passed = True
            score += 0.15
            if crop_supported:
                score += 0.05
        elif crop_ready:
            crop_status = "caution"
            crop_details = "Crop model artifacts are present, but the returned label is still a runtime placeholder."
            issues.append("Crop runtime is not returning a concrete label yet.")
            crop_passed = False
        else:
            issues.append("Crop model runtime is not connected yet.")

        checks.append(
            VerificationCheck(
                name="crop_prediction",
                status=crop_status,
                details=crop_details,
                passed=crop_passed,
            )
        )

        route_found = bool(disease_route.get("route_found", False))
        route_status = "passed" if route_found else "caution"
        route_details = (
            f"Disease routing selected `{disease_route.get('resolved_model_key', '')}` using `{disease_route.get('strategy', 'no_route')}`."
            if route_found
            else "No disease route could be verified for the current crop label."
        )
        checks.append(
            VerificationCheck(
                name="disease_routing",
                status=route_status,
                details=route_details,
                passed=route_found,
            )
        )
        if route_found:
            score += 0.10
        else:
            issues.append("Disease route is not confirmed yet.")

        disease_label = self._normalize_label(disease_prediction.get("label"))
        disease_ready = bool(disease_prediction.get("model_ready", False))
        disease_supported = self._label_in_supported_labels(
            disease_prediction.get("label"),
            disease_prediction.get("supported_labels", []),
        )
        disease_status = "pending"
        disease_details = "Disease model runtime is not connected yet."
        disease_passed: bool | None = None
        if disease_ready and disease_label and disease_label not in self._placeholder_tokens:
            disease_status = "passed"
            disease_details = "Disease prediction produced a concrete label."
            disease_passed = True
            score += 0.15
            if disease_supported:
                score += 0.05
        elif disease_ready:
            disease_status = "caution"
            disease_details = "Disease model artifacts are present, but the returned label is still a runtime placeholder."
            issues.append("Disease runtime is not returning a concrete label yet.")
            disease_passed = False
        else:
            issues.append("Disease model runtime is not connected yet.")

        checks.append(
            VerificationCheck(
                name="disease_prediction",
                status=disease_status,
                details=disease_details,
                passed=disease_passed,
            )
        )

        top_document = self._top_document(rag)
        rag_ready = bool(rag.get("knowledge_base_ready", False))
        rag_status = "pending"
        rag_details = "No knowledge-base evidence is available yet."
        rag_passed: bool | None = None

        if rag_ready and top_document is not None:
            rag_status = "passed"
            rag_details = f"Top retrieved document is `{top_document.get('title', '')}`."
            rag_passed = True
            score += 0.10
            matched_evidence_count = len(rag.get("documents", []))
        elif rag_ready:
            rag_status = "caution"
            rag_details = "Knowledge assets are loaded, but this request did not retrieve evidence."
            rag_passed = False
            issues.append("No retrieval evidence was returned for this request.")
        else:
            if is_healthy is not True:
                issues.append("RAG knowledge base is not connected yet.")

        checks.append(
            VerificationCheck(
                name="rag_retrieval",
                status=rag_status,
                details=rag_details,
                passed=rag_passed,
            )
        )

        if top_document is not None and disease_label not in self._placeholder_tokens:
            metadata = top_document.get("metadata", {})
            doc_crop = self._normalize_label(str(metadata.get("crop", "")))
            doc_disease = self._normalize_label(str(metadata.get("disease", "")))
            alias_values = {
                self._normalize_label(str(alias))
                for alias in metadata.get("aliases", [])
                if str(alias).strip()
            }

            crop_match = bool(crop_label and doc_crop == crop_label)
            disease_match = bool(
                disease_label
                and (
                    doc_disease == disease_label
                    or disease_label in alias_values
                )
            )

            evidence_status = "passed" if crop_match and disease_match else "caution"
            evidence_details = (
                "The top retrieved document aligns with the predicted crop and disease."
                if crop_match and disease_match
                else "The retrieved evidence does not fully align with the predicted crop and disease."
            )
            checks.append(
                VerificationCheck(
                    name="evidence_alignment",
                    status=evidence_status,
                    details=evidence_details,
                    passed=crop_match and disease_match,
                )
            )
            if crop_match:
                score += 0.05
            if disease_match:
                score += 0.10
            if crop_match and disease_match:
                matched_evidence_count = max(matched_evidence_count, 1)
            else:
                issues.append("Top retrieved evidence does not fully match the current prediction.")
        elif top_document is not None:
            checks.append(
                VerificationCheck(
                    name="evidence_alignment",
                    status="pending",
                    details="Evidence was retrieved, but prediction labels are still placeholders.",
                    passed=None,
                )
            )

        if crop_ready and disease_ready and route_found and rag_ready and matched_evidence_count > 0:
            status = "verified"
            verified = True
            summary = "Prediction checks, disease routing, and retrieved evidence are aligned."
            recommended_action = "You can present the retrieved disease guidance, while still advising expert confirmation for treatment decisions."
        elif crop_ready or disease_ready or rag_ready:
            status = "caution"
            verified = False
            summary = "Some verification signals are available, but the prediction is not fully backed by runtime outputs and aligned evidence yet."
            recommended_action = "Treat the output as provisional and keep expert confirmation language in the final answer."
        else:
            status = "awaiting_runtime"
            verified = False
            summary = "Verification is waiting for live crop/disease inference and aligned retrieval evidence."
            recommended_action = "Finish wiring the model runtime and provide label maps so the verifier can check real predictions."

        return VerificationResult(
            status=status,
            verified=verified,
            score=round(min(score, 1.0), 4),
            summary=summary,
            recommended_action=recommended_action,
            issues=list(dict.fromkeys(issues)),
            matched_evidence_count=matched_evidence_count,
            checks=checks,
        )

    def _top_document(self, rag_payload: dict[str, Any]) -> dict[str, Any] | None:
        documents = rag_payload.get("documents", [])
        if isinstance(documents, list) and documents:
            return documents[0]
        return None

    def _label_in_supported_labels(
        self,
        label: str | None,
        supported_labels: list[str],
    ) -> bool:
        normalized_label = self._normalize_label(label or "")
        if not normalized_label or normalized_label in self._placeholder_tokens:
            return False

        return normalized_label in {
            self._normalize_label(item)
            for item in supported_labels
            if str(item).strip()
        }

    @staticmethod
    def _normalize_label(value: str | None) -> str:
        text = str(value or "").strip().lower()
        return re.sub(r"[^a-z0-9]+", "_", text).strip("_")
