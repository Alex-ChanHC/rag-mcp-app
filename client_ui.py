# client_ui.py
import argparse
import asyncio
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
    def __init__(self, model: str = "qwen3:1.7b"):
        """
        Initializes the CombinedClient.

        Args:
            model (str): The Ollama model to use for processing queries.
                         Defaults to "qwen3:1.7b" for Qwen model support.
        """
        self.console = Console()
        self.exit_stack = AsyncExitStack()
        self.model = model
        self.ollama = ollama.AsyncClient() # Using ollama's async client
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict[str, Any]] = [] # To store tool metadata

    async def connect_to_server(self, server_script_path: str):
        """
        Establishes a connection to the MCP server.

        It determines the server's execution command based on the script's file extension
        (.py for Python) and sets up the communication transport.

        Args:
            server_script_path (str): The path to the server script file.

        Raises:
            ValueError: If the server script is not a .py file.
        """
        is_python = server_script_path.endswith('.py')
        if not is_python:
            raise ValueError("Server script must be a .py file")

        # Determine the command to run the server script
        command = "python"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

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

        It sends the query to the Ollama model, checks for tool calls in the response,
        executes any required tools, and then sends the tool results back to the model
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
        resp: ChatResponse = await self.ollama.chat(model=self.model, messages=messages, tools=self.tools)

        final_response_parts = []
        # Handle tool calls
        if resp.message.tool_calls:
            for tc in resp.message.tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments

                # Call the tool specified by the LLM
                # We need to map the tool name to the actual tool call function in the session
                # For simplicity, we assume the tool names directly map to session methods or we can use call_tool
                try:
                    # Use call_tool for generic tool execution
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    tool_result_content = tool_result.content[0].text if tool_result.content else ""
                    
                    # Append the tool's result to the messages for the next LLM call
                    messages.append({"role": "tool", "name": tool_name, "content": tool_result_content})

                    # Call the LLM again with the tool result to get the final answer
                    resp = await self.ollama.chat(model=self.model, messages=messages, tools=self.tools, options={"max_tokens": 500})
                    
                    # Append the final content from the LLM
                    if getattr(resp.message, "content", None):
                        final_response_parts.append(resp.message.content)
                except Exception as e:
                    self.console.print(f"Error calling tool {tool_name}: {e}", style="bold red")
                    messages.append({"role": "tool", "name": tool_name, "content": f"Error: {e}"})
                    # Continue to next tool call or final response if possible
                    resp = await self.ollama.chat(model=self.model, messages=messages, tools=self.tools, options={"max_tokens": 500})
                    if getattr(resp.message, "content", None):
                        final_response_parts.append(resp.message.content)

        # If no tool calls, or after tool calls, check for direct content
        if getattr(resp.message, "content", None):
            final_response_parts.append(resp.message.content)

        return "".join(final_response_parts)

    async def chat_loop(self):
        """
        Starts the interactive chat loop for the client UI.
        It uses Gradio to provide a web interface for user interaction.
        """
        self.console.print(f"[bold green]MCP Client UI Started![/bold green] [cyan]Model: {self.model}[/cyan]")
        self.console.print("[yellow]Access the UI in your browser.[/yellow]")

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
            textbox=gr.Textbox(
                placeholder="Ask me anything...",
                container=False,
                autoscroll=True,
                scale=7
            ),
            title="RAG MCP Chatbot",
            description="Interact with the RAG MCP server. Ask about weather or general knowledge.",
            theme="soft" # Example theme
        )

        # Launch the Gradio app
        # Use share=True to get a public URL if needed, but for local use, it's not necessary.
        # We need to run this in a way that doesn't block the main async loop if we were to do more.
        # For now, launching it directly is fine as it's the main interaction point.
        await asyncio.to_thread(chatbot_interface.launch)


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
    p.add_argument("--model", default="qwen3:1.7b", help="Ollama model to use for orchestration (default: qwen3:1.7b)")
    args = p.parse_args()

    # Instantiate the Combined client with the specified model
    client = CombinedClient(model=args.model)
    try:
        # Connect to the server
        await client.connect_to_server(args.mcp_server)
        # Start the chat UI
        await client.chat_loop()
    finally:
        # Ensure cleanup is called
        await client.cleanup()

if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main())
