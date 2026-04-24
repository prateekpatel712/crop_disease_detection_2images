from __future__ import annotations


class ResponseBuilder:
    _placeholder_labels = {
        "",
        "unknown",
        "model_not_uploaded",
        "runtime_not_connected",
        "not_available",
    }

    def build(self, state: dict) -> str:
        verification = state.get("verification", {})
        disease_prediction = state.get("disease_prediction", {})
        rag = state.get("rag", {})

        status = verification.get("status", "pending")

        if status in {"blocked_input", "blocked_quality", "blocked_non_leaf"}:
            return verification.get(
                "summary",
                "Please upload a clear crop leaf image so the diagnosis pipeline can run.",
            )

        lines: list[str] = []
        disease_label = disease_prediction.get("label")
        if self._has_concrete_label(disease_label):
            lines.append(f"Predicted disease: {disease_label}.")
        elif state.get("is_healthy") is True:
            lines.append("The uploaded leaf appears healthy.")
        else:
            lines.append(
                disease_prediction.get(
                    "note",
                    "The system could not produce a concrete disease label for this image.",
                )
            )

        lines.extend(self._build_rag_lines(rag))

        if len(lines) == 1 and status == "caution":
            lines.append(
                verification.get(
                    "recommended_action",
                    "Please retry with a clearer image if you want a stronger diagnosis.",
                )
            )

        return " ".join(item for item in lines if item).strip()

    def _build_rag_lines(self, rag: dict) -> list[str]:
        lines: list[str] = []
        documents = rag.get("documents", [])

        if not rag.get("knowledge_base_ready", False):
            return lines

        if not documents:
            return lines

        top_document = documents[0]
        metadata = top_document.get("metadata", {})
        summary = str(metadata.get("summary", "")).strip()
        symptoms = self._format_items(metadata.get("symptoms", []))
        causes = self._format_items(metadata.get("causes", []), limit=2)
        treatment = self._format_items(metadata.get("treatment", []))
        precautions = self._format_items(metadata.get("precautions", []))
        prevention = self._format_items(metadata.get("prevention", []))
        notes = str(metadata.get("notes", "")).strip()

        if summary:
            lines.append(summary)
        if symptoms:
            lines.append(f"Symptoms: {symptoms}.")
        if causes:
            lines.append(f"Causes: {causes}.")
        if treatment:
            lines.append(f"Treatment: {treatment}.")
        if precautions:
            lines.append(f"Precautions: {precautions}.")
        if prevention:
            lines.append(f"Prevention: {prevention}.")
        if notes:
            lines.append(f"Notes: {notes}.")

        return lines

    def _has_concrete_label(self, label: str | None) -> bool:
        return str(label or "").strip() not in self._placeholder_labels

    @staticmethod
    def _format_items(values: list[str], *, limit: int = 3) -> str:
        cleaned = [str(item).strip() for item in values if str(item).strip()]
        return ", ".join(cleaned[:limit])
