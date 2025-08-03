# client_ui.py
import argparse
import asyncio
import os
from contextlib import AsyncExitStack
import ollama
from ollama import ChatResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.markdown import Markdown
from typing import Optional, List, Dict, Any

# Gradio for the UI
import gradio as gr

# Load environment variables (e.g., for OpenAI API key if needed by client, though not directly used here)
from dotenv import load_dotenv
load_dotenv()

class CombinedClient:
    """
    A client application that connects to an MCP server, discovers its tools,
    and processes user queries using an LLM (Ollama) to interact with those tools.
    This client also includes a Gradio UI for interaction.
    """
    def __init__(self, orchestrator_model: str = "qwen3:1.7b", rag_llm_provider: str = "ollama"):
        """
        Initializes the CombinedClient.

        Args:
            orchestrator_model (str): The Ollama model to use for processing queries (orchestrator LLM).
                                      Defaults to "qwen3:1.7b" for Qwen model support.
            rag_llm_provider (str): The LLM provider for the RAG server ('ollama' or 'gemini').
                                    Defaults to 'ollama'.
        """
        self.console = Console()
        self.exit_stack = AsyncExitStack()
        self.orchestrator_model = orchestrator_model
        self.rag_llm_provider = rag_llm_provider
        self.ollama = ollama.AsyncClient() # Using ollama's async client for orchestrator
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict[str, Any]] = [] # To store tool metadata

    async def connect_to_server(self, server_script_path: str):
        """
        Establishes a connection to the MCP server.

        It determines the server's execution command based on the script's file extension
        (.py for Python) and sets up the communication transport, including passing
        the RAG LLM provider argument.

        Args:
            server_script_path (str): The path to the server script file.

        Raises:
            ValueError: If the server script is not a .py file.
        """
        is_python = server_script_path.endswith('.py')
        if not is_python:
            raise ValueError("Server script must be a .py file")

        # Determine the command to run the server script and pass the LLM provider argument
        command = "uv" # Use uv to run the script
        server_args = ["run", "python", server_script_path, f"--llm-provider={self.rag_llm_provider}"]
        server_params = StdioServerParameters(command=command, args=server_args, env=None)

        # Set up the standard I/O transport for the client-server communication
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        # Initialize the MCP client session
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # List available tools provided by the server
        meta = await self.session.list_tools()
        self.tools = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}} for t in meta.tools]
        tool_names = [t['function']['name'] for t in self.tools]
        self.console.print(f"[bold green]Server connected.[/bold green] Tools: {tool_names}")

    async def process_query_with_tools(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Processes a user query by interacting with the LLM and executing tools.

        It sends the query to the Ollama model (orchestrator), checks for tool calls in the response,
        executes any required tools via the MCP session, and then sends the tool results back to the model
        to get a final answer.

        Args:
            query (str): The user's input query.
            history (List[Dict[str, str]]): The conversation history.

        Returns:
            str: The final response from the LLM after processing the query and tool results.
        """
        # Prepare the initial message for the LLM, including history
        messages = []
        for msg in history:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": query})

        # Initial call to the LLM with tool definitions
        print(f"DEBUG: Messages before first LLM call: {messages}")
        resp: ChatResponse = await self.ollama.chat(model=self.orchestrator_model, messages=messages, tools=self.tools)
        print(f"DEBUG: First LLM response message: {resp.message}")

        final_response_parts = []
        # Handle tool calls
        if resp.message.tool_calls:
            for tc in resp.message.tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments

                # Call the tool specified by the LLM
                try:
                    # Use call_tool for generic tool execution
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    tool_result_content = tool_result.content[0].text if tool_result.content else ""
                    
                    # Append the tool's result to the messages for the next LLM call
                    messages.append({"role": "tool", "name": tool_name, "content": tool_result_content})

                    # Call the LLM again with the tool result to get the final answer
                    resp = await self.ollama.chat(model=self.orchestrator_model, messages=messages, tools=self.tools, options={"max_tokens": 500})
                    
                    # Append the final content from the LLM
                    if getattr(resp.message, "content", None):
                        final_response_parts.append(resp.message.content)
                except Exception as e:
                    self.console.print(f"Error calling tool {tool_name}: {e}", style="bold red")
                    messages.append({"role": "tool", "name": tool_name, "content": f"Error: {e}"})
                    # Continue to next tool call or final response if possible
                    resp = await self.ollama.chat(model=self.orchestrator_model, messages=messages, tools=self.tools, options={"max_tokens": 500})
                    if getattr(resp.message, "content", None):
                        final_response_parts.append(resp.message.content)

        # If no tool calls, or after tool calls, check for direct content
        if getattr(resp.message, "content", None):
            final_response_parts.append(resp.message.content)

        return "".join(final_response_parts)

    async def chat_loop(self, port: int): # Accept port as argument
        """
        Starts the interactive chat loop for the client UI.
        It uses Gradio to provide a web interface for user interaction.
        """
        self.console.print(f"[bold green]MCP Client UI Started![/bold green] [cyan]Orchestrator Model: {self.orchestrator_model}[/cyan]")
        self.console.print(f"[cyan]RAG Server LLM Provider: {self.rag_llm_provider}[/cyan]")
        self.console.print(f"[yellow]Access the UI in your browser on port {port}.[/yellow]") # Inform user about port

        # Gradio interface setup
        async def gradio_stream_response(message, history):
            # Convert Gradio history format to the format expected by process_query_with_tools
            formatted_history = []
            for human, ai in history:
                formatted_history.append({"role": "user", "content": human})
                formatted_history.append({"role": "assistant", "content": ai})
            
            # Process the query using the LLM and tools
            response = await self.process_query_with_tools(message, formatted_history)
            return response

        # Initiate the Gradio chat interface
        chatbot_interface = gr.ChatInterface(
            gradio_stream_response,
            chatbot=gr.Chatbot(type="messages"),
            textbox=gr.Textbox(
                placeholder="Ask me anything...",
                container=False,
                autoscroll=True,
                scale=7
            ),
            title="RAG MCP Chatbot",
            description=f"Interact with the RAG MCP server. RAG LLM: {self.rag_llm_provider.capitalize()}. Orchestrator LLM: {self.orchestrator_model}.",
            theme="soft", # Example theme
            stop_btn="Stop" # Add stop button
        )

        # Explicitly initialize stop_event for the Gradio interface
        chatbot_interface.stop_event = asyncio.Event()

        # Launch the Gradio app
        # Use share=True to get a public URL if needed, but for local use, it's not necessary.
        # We need to run this in a way that doesn't block the main async loop if we were to do more.
        # For now, launching it directly is fine as it's the main interaction point.
        await asyncio.to_thread(chatbot_interface.launch, server_port=port) # Use the passed port


    async def cleanup(self):
        """
        Cleans up resources by closing the asynchronous exit stack.
        This ensures all managed asynchronous resources are properly closed.
        """
        await self.exit_stack.aclose()

# Main function to parse arguments and run the client UI
async def main():
    """
    Parses command-line arguments and initializes the CombinedClient.
    It connects to the specified MCP server and starts the Gradio chat UI.
    Ensures cleanup is performed even if errors occur.
    """
    # Set up argument parser
    p = argparse.ArgumentParser()
    p.add_argument("--mcp-server", required=True, help="Path to the MCP server script (e.g., rag_server.py)")
    p.add_argument("--orchestrator-model", default=os.environ.get("ORCHESTRATOR_MODEL", "qwen3:1.7b"), help="Ollama model to use for orchestration (default: qwen3:1.7b)")
    p.add_argument("--rag-llm-provider", default=os.environ.get("RAG_LLM_PROVIDER", "ollama"), choices=["ollama", "gemini"],
                   help="LLM provider for the RAG server: 'ollama' or 'gemini' (default: ollama).")
    p.add_argument("--port", default=os.environ.get("GRADIO_PORT", "3000"), help="Port for the Gradio UI (default: 3000).")
    args = p.parse_args()

    # Instantiate the Combined client with the specified models and provider
    client = CombinedClient(orchestrator_model=args.orchestrator_model, rag_llm_provider=args.rag_llm_provider)
    try:
        # Connect to the server
        await client.connect_to_server(args.mcp_server)
        # Start the chat UI
        await client.chat_loop(port=int(args.port))
    finally:
        # Ensure cleanup is called
        await client.cleanup()

if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main())
