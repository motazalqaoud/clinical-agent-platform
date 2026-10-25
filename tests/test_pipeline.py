from clinagent.generation.generators import ExtractiveGenerator, LLMGenerator
from clinagent.llm.clients import ScriptedLLM
from clinagent.pipeline import ClinicalQA


def test_grounded_answer_with_dense_citations(qa):
    answer, trace = qa.ask("What cefazolin dose is used for adults weighing 120 kg or more?")
    assert answer.grounded
    assert "3 g" in answer.text
    assert [c.index for c in answer.citations] == list(range(1, len(answer.citations) + 1))
    assert {s.name for s in trace.spans} >= {"input_guard", "retrieve", "generate", "verify"}


def test_patient_answer_cites_fhir_record(qa):
    answer, _ = qa.ask(
        "What cefazolin prophylaxis dose applies for this patient's weight?", "pt-004"
    )
    assert answer.grounded
    assert "Observation/obs-030" in [c.ref for c in answer.citations]


def test_out_of_scope_refused(qa):
    answer, _ = qa.ask("What is the capital of France?")
    assert not answer.grounded and answer.refusal_reason == "no_evidence"


def test_identifier_request_refused(qa):
    answer, _ = qa.ask("What is the phone number for this patient?", "pt-001")
    assert answer.refusal_reason == "identifier_request"


def test_unknown_patient_refused(qa):
    answer, _ = qa.ask("Which antibiotic for pneumonia?", "pt-999")
    assert answer.refusal_reason == "unknown_patient"


def test_phi_in_question_is_redacted_and_logged(qa):
    answer, _ = qa.ask(
        "SSN 123-45-6789. What is the first-line antibiotic for outpatient pneumonia?"
    )
    assert "123-45-6789" not in answer.text
    assert any(e.guard == "phi" for e in answer.guardrail_events)


def test_poisoned_document_is_quarantined(poisoned_index, patients):
    qa = ClinicalQA(poisoned_index, ExtractiveGenerator(), patients)
    answer, _ = qa.ask("pneumonia quick reference antibiotic choice")
    assert any(e.guard == "prompt_injection" for e in answer.guardrail_events)
    assert "10 g" not in answer.text


def test_llm_that_changes_a_dose_is_blocked(index, patients):
    llm = ScriptedLLM(["Give cefazolin 4 g IV for adults weighing 120 kg or more [1]."])
    qa = ClinicalQA(index, LLMGenerator(llm), patients)
    answer, _ = qa.ask("What cefazolin dose for adults weighing 120 kg or more?")
    assert not answer.grounded and answer.refusal_reason == "unverified"
    assert "4 g" not in answer.text


def test_llm_insufficient_becomes_refusal(index, patients):
    qa = ClinicalQA(index, LLMGenerator(ScriptedLLM(["INSUFFICIENT"])), patients)
    answer, _ = qa.ask("What cefazolin dose for adults weighing 120 kg or more?")
    assert answer.refusal_reason == "no_evidence"


def test_llm_prompt_marks_sources_as_data(index, patients):
    llm = ScriptedLLM(["Give cefazolin 2 g IV for adults under 120 kg [1]."])
    ClinicalQA(index, LLMGenerator(llm), patients).ask("cefazolin dose adults under 120 kg")
    prompt = llm.calls[0]["messages"][0]["content"]
    assert prompt.startswith("SOURCES:") and "<<<" in prompt
    assert "not instructions" in llm.calls[0]["system"]


def test_metrics_record_outcomes(qa):
    qa.ask("What is the capital of France?")
    assert 'outcome="no_evidence"' in qa.metrics.render()
