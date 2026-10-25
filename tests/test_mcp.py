import asyncio
import json

from clinagent.mcp_server import create_server


def _payload(result):
    # MCP 2.x returns structured content alongside text content.
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured is not None:
        return structured.get("result", structured)
    return json.loads(result.content[0].text)


def test_tools_are_listed(qa):
    tools = asyncio.run(create_server(qa).list_tools())
    names = {t.name for t in tools}
    assert names == {"ask_clinical_question", "search_guidelines", "get_patient_context"}


def test_ask_tool_returns_citations(qa):
    server = create_server(qa)
    result = asyncio.run(
        server.call_tool(
            "ask_clinical_question", {"question": "oxygen saturation target acute asthma"}
        )
    )
    payload = _payload(result)
    assert payload["grounded"] and payload["citations"]


def test_patient_context_tool_has_no_identifiers(qa):
    server = create_server(qa)
    result = asyncio.run(
        server.call_tool("get_patient_context", {"patient_id": "pt-001", "section": "all"})
    )
    assert "SYN-0001" not in json.dumps(_payload(result))
