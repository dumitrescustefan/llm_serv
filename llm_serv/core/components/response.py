import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from llm_serv.api import Model
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import TokenTracker
from llm_serv.structured_response.model import StructuredResponse


class LLMResponse(BaseModel):    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: LLMRequest    
    
    output: StructuredResponse | str | None = None        
    
    native_response_format_used: bool | None = None
    tokens: TokenTracker = Field(default_factory=TokenTracker)
    
    llm_model: Model | None = None

    start_time: float | None = None  # time.time() as fractions of a second
    end_time: float | None = None  # time.time() as fractions of a second
    total_duration: float | None = None  # time in seconds of the entire request, including retries (fractions included)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("output", mode="before")
    @classmethod
    def _deserialize_output(cls, output, info):
        """Deserialize the output field from JSON string back to StructuredResponse if needed."""
        if output is None:
            return None
        
        # Check if we have field data available and if response_model is set
        if isinstance(output, str) and info.data and 'request' in info.data:
            request = info.data['request']
            if hasattr(request, 'response_model') and request.response_model is not None:
                try:
                    # Try to deserialize JSON string back to StructuredResponse
                    return StructuredResponse.deserialize(output)
                except Exception:
                    # If deserialization fails, return as string
                    return output
    
        # For StructuredResponse instances and other types, return as-is
        return output

    @field_serializer("output", when_used="json")
    def _serialize_output(self, output):
        """Serialize the output field for JSON output."""
        if output is None:
            return None
        if isinstance(output, StructuredResponse):
            # Use the serialize method to convert to JSON string
            return output.serialize()
        # For strings and other types, return as-is
        return output

    @classmethod
    def from_request(cls, request: LLMRequest) -> "LLMResponse":
        response = LLMResponse(request=request)
        return response

    def rprint(self, subtitle: str | None = None):
        try:
            import json
            from enum import Enum

            from rich import print as rprint
            from llm_serv.conversation.role import Role
            from rich.console import Console
            from rich.json import JSON
            from rich.panel import Panel

            console = Console()

            # Custom JSON encoder to handle Enums and other non-serializable types
            class EnhancedJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Enum):
                        return obj.value
                    try:
                        # Try to convert to dict if it has a model_dump method
                        if hasattr(obj, "model_dump"):
                            return obj.model_dump(exclude_none=True)
                        # Try to convert to dict if it has a dict method
                        if hasattr(obj, "__dict__"):
                            return obj.__dict__
                    except:  # noqa: E722
                        pass
                    # Let the base class handle it or raise TypeError
                    return super().default(obj)

            # Prepare panel content
            content_parts = []
            
            # Add system message if present
            if self.request.conversation.system:
                content_parts.append(f"[bold dark_magenta][SYSTEM][/bold dark_magenta] [dark_magenta]{self.request.conversation.system}[/dark_magenta]")  # noqa: E501
            
            # Process conversation messages
            for message in self.request.conversation.messages:
                if message.role == Role.USER:
                    content_parts.append(f"[bold dark_blue][USER][/bold dark_blue] [dark_blue]{message.text}[/dark_blue]")
                elif message.role == Role.ASSISTANT:
                    content_parts.append(f"[bold dark_green][ASSISTANT][/bold dark_green] [dark_green]{message.text}[/dark_green]")

            # Add the final output
            content_parts.append("[bold bright_green][ASSISTANT - OUTPUT][/bold bright_green]")
            if isinstance(self.output, str):
                content_parts.append(f"[bright_green]{self.output}[/bright_green]")
            else:
                try:
                    # First convert the data to a JSON-serializable format using our custom encoder
                    data = str(self.output)
                        
                    # Convert to JSON string with our custom encoder that handles Enums
                    json_str = json.dumps(data, indent=2, cls=EnhancedJSONEncoder)
                    
                    # Use rich's console to directly print the formatted JSON
                    content_parts.append("[bright_green]")
                    
                    # Create a temporary console that outputs to a string
                    str_console = Console(width=100, file=None)
                    with str_console.capture() as capture:
                        str_console.print(JSON.from_data(json.loads(json_str)))
                    
                    # Add the captured output to our content
                    content_parts.append(capture.get())
                    content_parts.append("[/bright_green]")
                except Exception as exc:
                    content_parts.append(f"[bright_red]Error serializing output: {str(exc)}[/bright_red]")
                    content_parts.append(f"[bright_red]Output type: {type(self.output)}[/bright_red]")

            # Create panel title (stats line)
            title = ""
            if self.tokens:
                model_str = f"LLMRequest: {self.llm_model.provider.name}/{self.llm_model.name}"
                title = f"{model_str} | Time: {self.total_duration:.2f}s | Input/Output tokens: {self.tokens.input_tokens}/{self.tokens.completion_tokens} | Total tokens: {self.tokens.total_tokens}"  # noqa: E501

            # Print single panel with all content
            console.print(
                Panel(
                    "\n".join(content_parts),
                    title=title,
                    title_align="right",
                    border_style="magenta",
                    subtitle=subtitle,
                    subtitle_align="left",
                )
            )
        except Exception as e:
            # Fallback to basic printing if rich formatting fails
            try:
                from rich import print as rprint
                rprint(f"[bold red]Error in rprint method: {str(e)}[/bold red]")
                rprint("[yellow]Falling back to basic output:[/yellow]")

                # Print basic conversation info
                if (
                    hasattr(self, "request")
                    and self.request
                    and hasattr(self.request, "conversation")
                ):
                    if (
                        hasattr(self.request.conversation, "system")
                        and self.request.conversation.system
                    ):
                        rprint(
                            f"[dark_magenta]System: {self.request.conversation.system}[/dark_magenta]"
                        )

                    if hasattr(self.request.conversation, "messages"):
                        for msg in self.request.conversation.messages:
                            role = getattr(msg, "role", "unknown")
                            text = getattr(msg, "text", "no text")
                            rprint(f"[blue]{role}: {text}[/blue]")

                # Print output
                if hasattr(self, "output"):
                    rprint(f"[green]Output: {self.output}[/green]")

                # Print token info
                if hasattr(self, "tokens") and self.tokens:
                    rprint(
                        f"[cyan]Tokens: {self.tokens.total_tokens} (Input: {self.tokens.input_tokens}, Output: {self.tokens.completion_tokens})[/cyan]"  # noqa: E501
                    )
            except Exception as inner_e:
                # Last resort: plain print without any formatting
                print(f"Error in rprint fallback: {str(inner_e)}")
                print(f"Original error: {str(e)}")
                print("Output:", self.output)

