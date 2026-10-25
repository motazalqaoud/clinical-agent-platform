from clinagent.agent.react import ClinicalAgent
from clinagent.llm.clients import ScriptedLLM


def test_agent_gathers_evidence_and_answers_with_citations(index, patients):
    llm = ScriptedLLM(
        [
            "Thought: check weight\nAction: get_patient_record\n"
            'Action Input: {"patient_id": "pt-004", "section": "observations"}',
            'Thought: find dose\nAction: search_guidelines\nAction Input: {"query": "cefazolin dose 120 kg"}',
            "Thought: done\nFinal Answer: The patient weighs 128 kg [1]. Give cefazolin 3 g IV for "
            "adults weighing 120 kg or more [2].",
        ]
    )
    result = ClinicalAgent(llm, index, patients).run("Cefazolin dose for pt-004?")
    assert result.verified
    assert result.evidence[0].ref == "Observation/obs-030"
    assert [s.action for s in result.steps] == ["get_patient_record", "search_guidelines", None]


def test_agent_answer_with_wrong_dose_is_withheld(index, patients):
    llm = ScriptedLLM(
        [
            'Thought: search\nAction: search_guidelines\nAction Input: {"query": "cefazolin 120 kg"}',
            "Thought: done\nFinal Answer: Give cefazolin 5 g IV [1].",
        ]
    )
    result = ClinicalAgent(llm, index, patients).run("cefazolin dose?")
    assert result.answer is None and not result.verified


def test_agent_recovers_from_unknown_tool_and_bad_input(index, patients):
    llm = ScriptedLLM(
        [
            "Thought: x\nAction: delete_records\nAction Input: {}",
            'Thought: x\nAction: get_patient_record\nAction Input: {"patient_id": "pt-001", "section": "x"}',
            "Thought: give up\nFinal Answer: The evidence does not answer this.",
        ]
    )
    result = ClinicalAgent(llm, index, patients).run("q")
    assert "unknown tool" in (result.steps[0].observation or "")
    assert (result.steps[1].observation or "").startswith("error")


def test_agent_stops_at_step_budget(index):
    llm = ScriptedLLM(["Thought: hmm"] * 3)
    result = ClinicalAgent(llm, index, max_steps=3).run("q")
    assert result.answer is None and len(result.steps) == 3
