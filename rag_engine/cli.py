import sys
import click
import asyncio
from typing import Callable, AsyncGenerator, Any, Optional


class REPLManager:

    def __init__(
        self,
        query_callback: Callable[[str], Any],
        model_selection_callback: Optional[Callable[[str], None]] = None,
        ingest_callback: Optional[Callable[[str], Any]] = None,
        core_engine: Any = None,
    ):

        self.query_callback = query_callback
        self.model_selection_callback = model_selection_callback
        self.ingest_callback = ingest_callback
        self.core_engine = core_engine
        self.current_model = "local-llm"


    def show_help(self):

        click.secho("\n--- Available Commands ---", fg="cyan", bold=True)
        click.echo("/help          - Show this help menu")
        click.echo("/model <name>  - Switch target LLM model (e.g. local-llm, openai-llm, anthropic-llm)")
        click.echo("/ingest <path> - Ingest a file or directory of documents")
        click.echo("/dedup         - Remove duplicate chunks from the vector store")
        click.echo("/clean-db      - Wipe the entire vector store (all chunks and parents)")
        click.echo("/clear         - Clear terminal screen")
        click.echo("/exit          - Quit the application\n")


    def clear_screen(self):

        click.clear()
        click.secho("=== RAG Engine Interactive CLI ===", fg="green", bold=True)


    def set_model(self, model_name: str):

        self.current_model = model_name
        if self.model_selection_callback:

            self.model_selection_callback(model_name)
        click.secho(f"Switched model to: {model_name}", fg="yellow")


    async def start_interactive_loop(self):

        self.clear_screen()
        self.show_help()

        while True:

            try:

                loop = asyncio.get_running_loop()
                user_input = (await loop.run_in_executor(
                    None,
                    lambda: click.prompt(
                        click.style("\nUser >", fg="blue", bold=True),
                        prompt_suffix=" "
                    )
                )).strip()
                
                if not user_input:

                    continue

                if user_input.startswith("/"):

                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    
                    if cmd == "/exit":

                        click.secho("Goodbye!", fg="red")
                        break
                        
                    elif cmd == "/help":

                        self.show_help()
                        
                    elif cmd == "/clear":

                        self.clear_screen()
                        
                    elif cmd == "/model":

                        if len(parts) > 1:

                            self.set_model(parts[1])
                        else:

                            click.secho("Usage: /model <model_name>", fg="red")
                            
                    elif cmd == "/ingest":

                        if not self.ingest_callback:

                            click.secho("Ingestion is not supported/configured in this context.", fg="red")
                        elif len(parts) > 1:

                            path = parts[1]
                            click.secho(f"Ingesting: {path}...", fg="cyan")
                            try:

                                if asyncio.iscoroutinefunction(self.ingest_callback):

                                    await self.ingest_callback(path)
                                else:

                                    self.ingest_callback(path)
                                click.secho("Ingestion completed successfully!", fg="green")
                            except Exception as ex:

                                click.secho(f"Ingestion failed: {ex}", fg="red")
                        else:

                            click.secho("Usage: /ingest <file_or_dir_path>", fg="red")

                    elif cmd == "/dedup":

                        if not self.core_engine:

                            click.secho("Deduplication is not available in this context.", fg="red")
                        else:

                            click.secho("Deduplicating chunks...", fg="cyan")
                            result = self.core_engine.deduplicate()
                            click.secho(
                                f"Removed {result['removed']} duplicate chunks. "
                                f"{result['remaining']} chunks remaining.",
                                fg="green",
                            )

                    elif cmd == "/clean-db":

                        if not self.core_engine:

                            click.secho("Database management is not available in this context.", fg="red")
                        else:

                            click.secho(
                                "⚠️  WARNING: This will permanently delete all ingested documents!",
                                fg="red",
                                bold=True,
                            )
                            loop = asyncio.get_running_loop()
                            confirm = await loop.run_in_executor(
                                None,
                                lambda: click.confirm("Are you sure you want to clean the database?", default=False),
                            )
                            if confirm:
                                result = self.core_engine.clean_database()
                                click.secho(
                                    f"Database wiped: {result['cleared']} chunks removed.",
                                    fg="green",
                                )
                            else:
                                click.secho("Cancelled.", fg="yellow")

                    else:

                        click.secho(f"Unknown command: {cmd}", fg="red")
                        
                    continue

                # Run query callback
                click.secho("Engine > ", fg="green", bold=True, nl=False)
                
                # Check if callback is an async generator or a standard async function returning a generator
                if asyncio.iscoroutinefunction(self.query_callback):

                    async for chunk in self.query_callback(user_input):

                        click.echo(chunk, nl=False)
                        sys.stdout.flush()
                else:

                    res = self.query_callback(user_input)
                    # Check if result is async generator
                    if hasattr(res, "__anext__"):

                        async for chunk in res:

                            click.echo(chunk, nl=False)
                            sys.stdout.flush()
                    elif hasattr(res, "__iter__") or hasattr(res, "__next__"):

                        for chunk in res:

                            click.echo(chunk, nl=False)
                            sys.stdout.flush()
                    else:

                        click.echo(res, nl=False)
                        
                click.echo()

            except (KeyboardInterrupt, EOFError):

                click.secho("\nGoodbye!", fg="red")
                break
                
            except Exception as e:

                click.secho(f"\nError encountered: {e}", fg="red")
