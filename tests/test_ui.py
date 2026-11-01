from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_home_renders_workflow_links():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == '📊 FinSight AI'
    assert len(app.get('page_link')) == 2


def test_document_page_local_inspection_without_key(monkeypatch, tmp_path):
    from src import config
    from dataclasses import replace
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setattr(config, 'load_settings', lambda *a, **k: replace(config.Settings(), storage_root=tmp_path))
    app = AppTest.from_file(str(ROOT / 'pages/1_Document_Assistant.py')).run(timeout=60)
    assert not app.exception
    app.radio[0].set_value('Fictional sample documents').run()
    next(button for button in app.button if button.label == 'Inspect locally (no API call)').click().run()
    assert not app.exception
    assert any('120 crore' in text.value for text in app.text)
    assert next(button for button in app.button if button.label == 'Index selected documents').disabled
    assert next(button for button in app.button if button.label == 'Ask Gemini').disabled


def test_portfolio_page_rerun_retention_and_stale_invalidation():
    app = AppTest.from_file(str(ROOT / 'pages/2_Portfolio_Optimizer.py')).run(timeout=60)
    assert not app.exception
    app.radio[0].set_value('Simulated example').run()
    next(button for button in app.button if button.label == 'Optimize portfolios').click().run(timeout=60)
    assert not app.exception and not app.error
    assert len(app.metric) == 6
    assert len(app.dataframe) == 1
    app.run()
    assert len(app.metric) == 6
    app.number_input[0].set_value(5.0).run()
    assert not app.exception and len(app.metric) == 0


def test_document_index_qa_and_reset_with_explicit_api_test_doubles(monkeypatch, tmp_path, embedder):
    """UI integration test ONLY: real Chroma, fake embeddings and chat; not live Gemini."""
    from dataclasses import replace
    from unittest.mock import Mock
    from src import config
    from src.rag.qa_chain import AnswerPayload
    monkeypatch.setattr(config, 'load_settings', lambda *a, **k: replace(config.Settings(api_key='unit-test-only'), storage_root=tmp_path))
    monkeypatch.setattr('src.rag.embeddings.GeminiEmbeddings', lambda _: embedder)
    monkeypatch.setattr('src.rag.embeddings.validate_models', lambda _: {})
    client = Mock()
    client.invoke.return_value = AnswerPayload(answer='Test-provider response: net profit is Rs. 15 crore [E1].', citation_ids=['E1'], insufficient=False)
    monkeypatch.setattr('src.rag.qa_chain.chat_client', lambda _: client)
    app = AppTest.from_file(str(ROOT / 'pages/1_Document_Assistant.py')).run(timeout=60)
    app.radio[0].set_value('Fictional sample documents').run()
    next(button for button in app.button if button.label == 'Index selected documents').click().run(timeout=60)
    assert not app.exception and not app.error
    assert len(app.session_state['selected_document_indexes']) == 1
    calls = embedder.document_calls
    app.text_area[0].set_value('What is net profit?').run()
    next(button for button in app.button if button.label == 'Ask Gemini').click().run(timeout=60)
    assert not app.exception and not app.error
    assert 'qa_result' in app.session_state
    assert any('net profit is Rs. 15 crore' in item.value for item in app.markdown)
    app.run()
    assert 'qa_result' in app.session_state and embedder.document_calls == calls
    app.text_area[0].set_value('Different question?').run()
    assert 'qa_result' not in app.session_state
    app.checkbox[0].check().run()
    next(button for button in app.button if button.label == 'Reset session indexes').click().run(timeout=60)
    assert not app.exception and not app.error
    assert app.session_state['selected_document_indexes'] == []
