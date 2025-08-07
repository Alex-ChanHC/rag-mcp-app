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
    def __init__(self, orchestrator_model: str = "qwen3:1.7b"):
        """
        Initializes the CombinedClient.

        Args:
            orchestrator_model (str): The Ollama model to use for processing queries (orchestrator LLM).
                                      Defaults to "qwen3:1.7b" for Qwen model support.
        """
        self.console = Console()
        self.exit_stack = AsyncExitStack()
        self.orchestrator_model = orchestrator_model
        self.ollama = ollama.AsyncClient() # Using ollama's async client for orchestrator
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
        command = "uv" # Use uv to run the script
        server_args = ["run", "python", server_script_path]
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

    async def process_query_with_tools(self, query: str, history: List[Dict[str, str]]) -> tuple[str, str]:
        """
        Processes a user query by interacting with the LLM and executing tools.
        This version handles a two-step RAG process internally and returns thoughts and response.
        """
        # Prepare the initial message for the LLM, including history
        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg.get("role"), "content": msg.get("content")})
        messages.append({"role": "user", "content": query})
        self.console.print("DEBUG: Messages sent to orchestrator LLM:", messages, style="bold blue")

        # 1. First LLM call to determine which tool to use (if any)
        self.console.print("DEBUG: First LLM call to select a tool.", style="yellow")
        resp: ChatResponse = await self.ollama.chat(
            model=self.orchestrator_model, messages=messages, tools=self.tools
        )
        self.console.print(f"DEBUG: First LLM response message: {resp.message}", style="yellow")

        # Extract thoughts from the initial response
        raw_content = resp.message.content or ""
        thoughts = ""
        if "<think>" in raw_content and "</think>" in raw_content:
            start = raw_content.find("<think>") + len("<think>")
            end = raw_content.find("</think>")
            thoughts = raw_content[start:end].strip()
        
        self.console.print(f"DEBUG: Extracted thoughts: {thoughts}", style="bold yellow")

        # Append the assistant's response (e.g., the tool call) to messages
        messages.append(resp.message)

        # 2. Handle tool calls, if any
        if resp.message.tool_calls:
            for tc in resp.message.tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments
                self.console.print(f"DEBUG: Calling tool '{tool_name}' with args: {tool_args}", style="cyan")

                try:
                    # Execute the tool via the MCP session
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    tool_result_content = tool_result.content[0].text if tool_result.content else ""
                    self.console.print(f"DEBUG: Tool '{tool_name}' result: {tool_result_content}", style="cyan")
                    
                    # Append the tool's result to the messages for the next LLM call
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": tc.id,
                        "name": tool_name, 
                        "content": tool_result_content
                    })

                except Exception as e:
                    self.console.print(f"Error calling tool {tool_name}: {e}", style="bold red")
                    messages.append({"role": "tool", "name": tool_name, "content": f"Error: {e}"})

            # 3. Second LLM call to generate a final answer using the tool's output
            self.console.print("DEBUG: Second LLM call to generate final response from tool output.", style="yellow")
            final_resp: ChatResponse = await self.ollama.chat(
                model=self.orchestrator_model, messages=messages
            )
            final_content = final_resp.message.content
            self.console.print(f"DEBUG: Final LLM response: {final_content}", style="bold green")
            return thoughts, final_content

        # If no tool was called, return the direct response from the first call
        final_content = raw_content.split("</think>", 1)[1].strip() if thoughts else raw_content
        self.console.print(f"DEBUG: No tool called. Final response: {final_content}", style="bold green")
        return thoughts, final_content

    async def chat_loop(self, port: int): # Accept port as argument
        """
        Starts the interactive chat loop for the client UI.
        It uses Gradio to provide a web interface for user interaction.
        """
        self.console.print(f"[bold green]MCP Client UI Started![/bold green] [cyan]Orchestrator Model: {self.orchestrator_model}[/cyan]")
        self.console.print(f"[yellow]Access the UI in your browser on port {port}.[/yellow]") # Inform user about port

        # Gradio interface setup
        async def gradio_stream_response(message, history):
            self.console.print(f"DEBUG: Gradio received message: '{message}'", style="bold magenta")
            self.console.print("DEBUG: Gradio history object:", history, style="bold magenta")
            
            # The history is already in the correct format of a list of dicts.
            # We append the new user message to it.
            history.append({"role": "user", "content": message})
            
            thoughts, response = await self.process_query_with_tools(message, history)
            self.console.print(f"DEBUG: Final response to be streamed to Gradio: '{response}'", style="bold green")

            # Append the initial empty assistant message
            history.append({"role": "assistant", "content": ""})

            # Stream the response word by word, updating the last message in the history
            buffer = ""
            for char in response:
                buffer += char
                history[-1]["content"] = buffer
                yield thoughts, history

        with gr.Blocks(theme="soft") as demo:
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(type="messages", label="Chat History")
                    textbox = gr.Textbox(
                        placeholder="Ask me anything...",
                        container=False,
                        autoscroll=True,
                        scale=7
                    )
                with gr.Column(scale=1):
                    thought_bubble = gr.Markdown("### Agent Thoughts\n---", label="Thought Bubble")

            # Set up the interaction
            chatbot.like(lambda: gr.Info("Thanks for the feedback!"), None, None)

            textbox.submit(
                gradio_stream_response,
                [textbox, chatbot],
                [thought_bubble, chatbot],
            ).then(lambda: gr.update(value=""), None, textbox, queue=False)

        # Launch the Gradio app
        # Use share=True to get a public URL if needed, but for local use, it's not necessary.
        # We need to run this in a way that doesn't block the main async loop if we were to do more.
        # For now, launching it directly is fine as it's the main interaction point.
        await asyncio.to_thread(demo.launch, server_port=port) # Use the passed port


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
    p.add_argument("--port", default=os.environ.get("GRADIO_PORT", "3000"), help="Port for the Gradio UI (default: 3000).")
    args = p.parse_args()

    # Instantiate the Combined client with the specified models and provider
    client = CombinedClient(orchestrator_model=args.orchestrator_model)
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
