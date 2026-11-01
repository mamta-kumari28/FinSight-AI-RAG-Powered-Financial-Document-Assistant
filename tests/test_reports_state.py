import json
from dataclasses import replace

import pandas as pd
import pytest

from src.config import Settings, load_settings
from src.portfolio.optimizer import Portfolio
from src.rag.qa_chain import Answer
from src.rag.vector_store import Evidence
from src.reporting.report_generator import allocation_csv, json_bytes, qa_report
from src.state import fingerprint, sync_inputs


def test_qa_report_has_question_evidence_sources_and_no_keys():
    answer = Answer('What was revenue?', 'Revenue was 120 [E1].', [{'id': 'E1', 'source': 'abc.pdf · page 1'}],
                    [Evidence('Revenue: 120', {'filename': 'abc.pdf', 'page': 1}, .1)], False)
    markdown, report = qa_report(answer, {'chat': 'configured-chat', 'embedding': 'configured-embed'})
    assert answer.question in markdown and 'page 1' in markdown and 'Revenue: 120' in markdown
    decoded = json.loads(json_bytes(report))
    assert decoded['retrieved_evidence'][0]['text'] == 'Revenue: 120'
    assert 'api_key' not in json.dumps(decoded)


def test_csv_formula_injection_protection():
    portfolio = Portfolio('Minimum variance', pd.Series([.4, .6], index=['=CMD()', 'Safe']), {}, 'CLARABEL', 'optimal')
    output = allocation_csv([portfolio]).decode()
    assert "'=CMD()" in output


def test_json_rejects_nonfinite_metrics():
    with pytest.raises(ValueError):
        json_bytes({'metric': float('nan')})


def test_result_persists_for_same_inputs_but_clears_when_changed():
    state = {}
    identity = fingerprint(b'upload', .04, 'source')
    assert sync_inputs(state, 'inputs', identity, ('result',))
    state['result'] = 'computed portfolio'
    assert not sync_inputs(state, 'inputs', identity, ('result',))
    assert state['result'] == 'computed portfolio'
    assert sync_inputs(state, 'inputs', fingerprint(b'new upload', .04, 'source'), ('result',))
    assert 'result' not in state


@pytest.mark.parametrize('change', ['document', 'model', 'question', 'selected-index'])
def test_qa_input_change_invalidates_answer(change):
    state = {}
    original = ['document', 'model', 'question', 'selected-index']
    sync_inputs(state, 'inputs', fingerprint(*original), ('qa_result',))
    state['qa_result'] = 'old answer'
    modified = [item + '-new' if item == change else item for item in original]
    sync_inputs(state, 'inputs', fingerprint(*modified), ('qa_result',))
    assert 'qa_result' not in state


def test_configuration_key_redacted_and_secure_mapping(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    settings = load_settings({'GEMINI_API_KEY': 'unit-test-secret'})
    assert settings.api_key == 'unit-test-secret'
    assert 'unit-test-secret' not in repr(settings)
