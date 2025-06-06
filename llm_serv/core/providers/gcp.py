import os
from google import genai

from llm_serv.api import Model
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import ModelTokens
from llm_serv.core.exceptions import (
    CredentialsException,
    InternalConversionException,
    ServiceCallException,
    ServiceCallThrottlingException,
)


class GoogleLLMProvider(LLMProvider):
    @staticmethod
    def check_credentials() -> None:
        """
        Check required Google Cloud environment variables for Vertex AI.
        Uses the Application Default Credentials (ADC) approach with required environment variables.
        """
        required_variables = [
            "GOOGLE_CLOUD_PROJECT", 
            "GOOGLE_CLOUD_LOCATION",
        ]
        
        missing_vars = []
        for var in required_variables:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise CredentialsException(
                f"Missing required environment variables for Google Vertex AI: {', '.join(missing_vars)}. "
                f"Set up authentication using gcloud CLI or service account key."
            )

    def __init__(self, model: Model):
        super().__init__(model)
        
        GoogleLLMProvider.check_credentials()
        
        self._project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self._location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self._client = None

    async def start(self):
        """
        Initialize the Google GenAI client for Vertex AI.
        """
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=self._project_id,
                location=self._location
            )

    async def stop(self):
        """
        Clean up the Google GenAI client.
        """
        if self._client:
            # Google GenAI client doesn't require explicit cleanup
            self._client = None

    async def _convert(self, request: LLMRequest) -> dict:
        """
        Convert internal LLMRequest to Google GenAI format.
        
        Google GenAI format supports:
        - text content
        - image content (as base64 data URIs)
        - system instructions
        
        Returns a dictionary with 'contents' and 'config' for the API call.
        """
        try:
            contents = []
            system_instruction = None

            # Handle system message if present
            if request.conversation.system is not None and len(request.conversation.system) > 0:
                system_instruction = request.conversation.system

            # Process each message in the conversation
            for message in request.conversation.messages:
                # Google GenAI role mapping
                if message.role == Role.USER:
                    role = "user"
                elif message.role == Role.ASSISTANT:
                    role = "model"  # Google uses "model" instead of "assistant"
                else:
                    role = message.role.value

                parts = []

                # Add text content if present
                if message.text:
                    parts.append({"text": message.text})

                # Add images if present
                for image in message.images:
                    # Convert image to base64 data URI
                    image_data = {
                        "inline_data": {
                            "mime_type": f"image/{image.format or 'jpeg'}",
                            "data": image.export_as_base64(image.image)
                        }
                    }
                    parts.append(image_data)

                if parts:  # Only add message if it has content
                    contents.append({
                        "role": role,
                        "parts": parts
                    })

            # Configuration for the generation
            config = {
                "max_output_tokens": request.max_completion_tokens if request.max_completion_tokens is not None else self.model.max_output_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }

            # Remove None values from config
            config = {k: v for k, v in config.items() if v is not None}

            return {
                "contents": contents,
                "system_instruction": system_instruction,
                "config": config
            }

        except Exception as e:
            raise InternalConversionException(f"Failed to convert request for Google Vertex AI: {str(e)}") from e

    async def _llm_service_call(self, request: LLMRequest) -> tuple[str, ModelTokens]:
        """
        Make a call to Google Vertex AI using the unified Google GenAI SDK.
        Returns a tuple of (output_text, tokens_info)
        """
        # Prepare request
        try:
            processed = await self._convert(request)
            contents = processed["contents"]
            system_instruction = processed["system_instruction"]
            config = processed["config"]
        except Exception as e:
            raise InternalConversionException(f"Failed to convert request: {str(e)}") from e

        # Call the Google GenAI service
        try:
            # Prepare parameters for the API call
            generate_params = {
                "model": self.model.internal_model_id,
                "contents": contents,
                "config": config,
            }

            # Add system instruction if present
            if system_instruction:
                generate_params["system_instruction"] = system_instruction

            # Generate content using the unified Google GenAI SDK
            response = await self._client.aio.models.generate_content(**generate_params)
            
            # Extract output text
            if not response.candidates or not response.candidates[0].content.parts:
                raise ServiceCallException("Google Vertex AI returned empty response")
            
            output = response.candidates[0].content.parts[0].text

            # Extract token usage information
            usage = response.usage_metadata
            tokens = ModelTokens(
                input_tokens=getattr(usage, 'prompt_token_count', 0),
                output_tokens=getattr(usage, 'candidates_token_count', 0),
                total_tokens=getattr(usage, 'total_token_count', 0),
            )

        except Exception as e:
            # Check for throttling/rate limiting errors
            error_message = str(e).lower()
            if any(phrase in error_message for phrase in ['rate limit', 'quota', 'throttle', 'too many requests', '429']):
                raise ServiceCallThrottlingException(f"Google Vertex AI service is throttling requests: {str(e)}") from e
            
            # Check for other known error patterns
            if any(phrase in error_message for phrase in ['permission denied', 'unauthorized', '401', '403']):
                raise ServiceCallException(f"Google Vertex AI authentication error: {str(e)}") from e

            # General service error
            raise ServiceCallException(f"Google Vertex AI service error: {str(e)}") from e

        return output, tokens


if __name__ == "__main__":
    import asyncio
    from pydantic import Field
    from llm_serv import LLMService
    from llm_serv.conversation.conversation import Conversation
    from llm_serv.conversation.role import Role
    from llm_serv.structured_response.model import StructuredResponse

    async def test_google():
        """Test function for GoogleLLMProvider"""
        model = LLMService.get_model("GOOGLE/gemini-2.0-flash-exp")
        llm = GoogleLLMProvider(model)

        class MyClass(StructuredResponse):
            example_string: str = Field(
                default="", description="A string field that should be filled with a random person name in Elven language"
            )
            example_int: int = Field(
                default=0, ge=0, le=10, description="An integer field with a random value, greater than 5."
            )
            example_float: float = Field(
                default=0, ge=0.0, le=10.0, description="A float field with a value exactly half of the integer value"
            )

        conversation = Conversation.from_prompt("Please fill in the following class respecting the following instructions.")
        conversation.add_text_message(role=Role.USER, content=MyClass.to_text())

        request = LLMRequest(conversation=conversation, response_model=MyClass)

        response = await llm(request)
        
        print(response)
        assert isinstance(response.output, MyClass)
    
        await llm.stop()

    # Run the test function with asyncio
    asyncio.run(test_google())