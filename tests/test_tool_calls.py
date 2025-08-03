import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the necessary modules before they are imported by the application code
# This is crucial because the application code will import these modules, and we need our mocks to be in place first.
# We are mocking the entire ollama module
mock_ollama = MagicMock()
# We need to mock the instance of the client, not the class itself
mock_ollama_client = AsyncMock()
mock_ollama.AsyncClient.return_value = mock_ollama_client

# We are also mocking the ClientSession and stdio_client from mcp
mock_mcp_client_stdio = MagicMock()
mock_mcp_client = MagicMock(stdio=mock_mcp_client_stdio)
mock_mcp = MagicMock(ClientSession=AsyncMock(), client=mock_mcp_client)


# Apply the patches to the system's module cache
# This ensures that when client_ui imports these modules, it gets our mocks instead of the real ones
with patch.dict('sys.modules', {
    'ollama': mock_ollama,
    'mcp': mock_mcp,
    'mcp.client': mock_mcp.client,
    'mcp.client.stdio': mock_mcp.client.stdio,
}):
    from client_ui import CombinedClient

@pytest.fixture
def client():
    """
    Pytest fixture to create an instance of CombinedClient for testing.
    This fixture initializes the client with default parameters.
    """
    return CombinedClient(orchestrator_model="test-model", rag_llm_provider="test-provider")

@pytest.mark.asyncio
async def test_process_query_for_weather_tool(client):
    """
    Test case to verify that the 'get_weather' tool is called correctly.

    This test simulates a user query about weather and checks if the CombinedClient
    correctly identifies the intent, calls the 'get_weather' tool, and processes the response.
    """
    # Arrange: Set up the mock responses from the LLM and the tool
    # 1. Mock the initial LLM response to simulate a tool call for 'get_weather'
    mock_llm_response_with_tool_call = MagicMock()
    mock_llm_response_with_tool_call.message.tool_calls = [
        MagicMock(function=MagicMock(name="get_weather", arguments={"city": "London"}))
    ]
    # 2. Mock the final LLM response after the tool has been called
    mock_llm_response_final = MagicMock()
    mock_llm_response_final.message.content = "The weather in London is sunny."
    mock_llm_response_final.message.tool_calls = None # No further tool calls

    # Configure the async chat mock to return the two responses in sequence
    client.ollama.chat = AsyncMock(side_effect=[
        mock_llm_response_with_tool_call,
        mock_llm_response_final
    ])

    # 3. Mock the MCP session and the 'call_tool' method
    client.session = AsyncMock()
    mock_tool_result = MagicMock()
    mock_tool_result.content = [MagicMock(text="Sunny")]
    client.session.call_tool = AsyncMock(return_value=mock_tool_result)

    # Act: Process a query that should trigger the weather tool
    query = "What's the weather in London?"
    history = []
    final_response = await client.process_query_with_tools(query, history)

    # Assert: Verify the control flow
    # 1. Verify the initial LLM call and its response format
    assert client.ollama.chat.call_count == 2
    first_call_args, first_call_kwargs = client.ollama.chat.call_args_list[0]
    assert 'messages' in first_call_kwargs
    assert first_call_kwargs['messages'][-1]['role'] == 'user'
    assert first_call_kwargs['messages'][-1]['content'] == query

    # 2. Verify the MCP tool call
    client.session.call_tool.assert_called_once_with("get_weather", {"city": "London"})

    # 3. Verify the second LLM call includes the tool result
    second_call_args, second_call_kwargs = client.ollama.chat.call_args_list[1]
    assert 'messages' in second_call_kwargs
    assert second_call_kwargs['messages'][-1]['role'] == 'tool'
    assert second_call_kwargs['messages'][-1]['name'] == 'get_weather'
    assert second_call_kwargs['messages'][-1]['content'] == "Sunny"

    # 4. Verify that a final, non-empty response was generated
    assert isinstance(final_response, str)
    assert len(final_response) > 0

@pytest.mark.asyncio
async def test_process_query_for_rag_tool(client):
    """
    Test case to verify that the 'get_rag_response' tool is called correctly.

    This test simulates a user query that requires information from the RAG system
    and checks if the CombinedClient correctly calls the 'get_rag_response' tool.
    """
    # Arrange: Set up the mock responses
    # 1. Mock the LLM response to simulate a tool call for RAG
    mock_llm_response_with_tool_call = MagicMock()
    mock_llm_response_with_tool_call.message.tool_calls = [
        MagicMock(function=MagicMock(name="get_rag_response", arguments={"query": "What is attention?"}))
    ]
    # 2. Mock the final LLM response
    mock_llm_response_final = MagicMock()
    mock_llm_response_final.message.content = "Attention is a mechanism in neural networks."
    mock_llm_response_final.message.tool_calls = None

    # Configure the async chat mock
    client.ollama.chat = AsyncMock(side_effect=[
        mock_llm_response_with_tool_call,
        mock_llm_response_final
    ])

    # 3. Mock the MCP session and the tool result
    client.session = AsyncMock()
    mock_tool_result = MagicMock()
    mock_tool_result.content = [MagicMock(text="Attention is a mechanism...")]
    client.session.call_tool = AsyncMock(return_value=mock_tool_result)

    # Act: Process a query that should trigger the RAG tool
    query = "What is attention?"
    history = []
    final_response = await client.process_query_with_tools(query, history)

    # Assert: Verify the control flow
    # 1. Verify the initial LLM call and its response format
    assert client.ollama.chat.call_count == 2
    first_call_args, first_call_kwargs = client.ollama.chat.call_args_list[0]
    assert 'messages' in first_call_kwargs
    assert first_call_kwargs['messages'][-1]['role'] == 'user'
    assert first_call_kwargs['messages'][-1]['content'] == query

    # 2. Verify the MCP tool call
    client.session.call_tool.assert_called_once_with("get_rag_response", {"query": "What is attention?"})

    # 3. Verify the second LLM call includes the tool result
    second_call_args, second_call_kwargs = client.ollama.chat.call_args_list[1]
    assert 'messages' in second_call_kwargs
    assert second_call_kwargs['messages'][-1]['role'] == 'tool'
    assert second_call_kwargs['messages'][-1]['name'] == 'get_rag_response'
    assert second_call_kwargs['messages'][-1]['content'] == "Attention is a mechanism..."

    # 4. Verify that a final, non-empty response was generated
    assert isinstance(final_response, str)
    assert len(final_response) > 0
