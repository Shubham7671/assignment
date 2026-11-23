"""Lead Qualification & Readiness Assistant (Project 027).

An LMS/CRM workflow that helps counsellors decide whether an enquiry carries
enough information for a useful consultation.

Pipeline:
    read lead history -> NLP extraction -> completeness scoring
    -> LLM summary & follow-up questions -> counsellor review

The assistant assesses *information completeness only*. It never scores,
ranks or filters a person, and protected attributes are excluded from
scoring and redacted before anything is sent to the LLM.
"""

__version__ = "1.0.0"
__project__ = "Project 027 - Lead Qualification & Readiness Assistant"

__all__ = ["__version__", "__project__"]
