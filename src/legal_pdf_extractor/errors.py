
class LegalPdfExtractorError(Exception):
    code = "extractor_error"

    def to_error_payload(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


class PdfNormalizationError(LegalPdfExtractorError):
    code = "pdf_normalization_error"


class PdfParseError(LegalPdfExtractorError):
    code = "pdf_parse_error"


class LLMError(LegalPdfExtractorError):
    code = "llm_error"


class EmbeddingModelError(LegalPdfExtractorError):
    code = "embedding_model_error"


class TypeValidationError(LegalPdfExtractorError):
    code = "type_validation_error"


class SourceVerificationError(LegalPdfExtractorError):
    code = "source_verification_error"
