from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import _init_session_state, _dashboard_view


class MockSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def test_init_session_state():
    mock_state = MockSessionState()
    with patch("streamlit.session_state", mock_state):
        _init_session_state()
        assert mock_state.token is None
        assert mock_state.user is None
        assert mock_state.active_collection_id is None
        assert mock_state.active_session_id is None


@patch("streamlit.columns")
@patch("streamlit.metric")
@patch("streamlit.subheader")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.title")
@patch("streamlit.write")
@patch("streamlit.button", return_value=False)
@patch("api_client.ApiClient.list_collections")
@patch("api_client.ApiClient.list_documents")
@patch("api_client.ApiClient.list_sessions")
def test_dashboard_view_renders_metrics(
    mock_list_sessions,
    mock_list_docs,
    mock_list_cols,
    mock_button,
    mock_write,
    mock_title,
    mock_div,
    mock_md,
    mock_sub,
    mock_metric,
    mock_cols,
):
    col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
    mock_cols.return_value = (col1, col2, col3)
    mock_list_cols.return_value = [{"id": "c1"}, {"id": "c2"}]
    mock_list_docs.return_value = [{"id": "d1"}]
    mock_list_sessions.return_value = [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}]

    mock_state = MockSessionState(user={"email": "alice@example.com", "full_name": "Alice"}, token="mock-token")
    with patch("streamlit.session_state", mock_state):
        _dashboard_view()

    col1.metric.assert_called_once_with("Collections", 2)
    col2.metric.assert_called_once_with("Documents", 1)
    col3.metric.assert_called_once_with("Conversations", 3)
