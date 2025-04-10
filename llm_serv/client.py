import httpx
from rich import print as rprint

from llm_serv.exceptions import InternalConversionException, ModelNotFoundException, ServiceCallException, ServiceCallThrottlingException, StructuredResponseException, TimeoutException, CredentialsException
from llm_serv.providers.base import LLMRequest, LLMResponse, LLMResponseFormat


class LLMServiceClient:
    def __init__(self, host: str, port: int, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = self._validate_timeout(timeout)
        
        self.provider = None
        self.name = None
        
        # Default headers to accept gzip compression
        self._default_headers = {
            "Accept-Encoding": "gzip, deflate",
            "Content-Type": "application/json"
        }
        
        # Create a persistent client for multiple requests
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            headers=self._default_headers
        )

    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _validate_timeout(self, timeout: float) -> float:
        """
        Enforce a minimum timeout of 5 seconds.
        
        Args:
            timeout: The timeout value in seconds to validate
            
        Returns:
            float: The validated timeout (minimum 5 seconds)
        """
        return 5 if timeout <= 0 else timeout

    async def server_health_check(self, timeout: float = 5.0) -> None:
        """
        Performs a health check by calling the /health endpoint of the server.
        Should be called immediately after construction when in an async context.
        It does NOT test models, only the server itself.
        
        Args:
            timeout: Maximum time to wait for health check response in seconds
            
        Raises:
            ServiceCallException: If the server is not healthy or cannot be reached
        """
        timeout = self._validate_timeout(timeout)
        
        try:
            # Use a temporary client with a short timeout for health check
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers=self._default_headers
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("detail", str(error_data))
                    raise ServiceCallException(f"Server health check failed: {error_msg}")
                    
                health_data = response.json()
                if health_data.get("status") != "healthy":
                    raise ServiceCallException(f"Server reported unhealthy status: {health_data}")
                    
        except httpx.TimeoutException:
            raise ServiceCallException(f"Health check timed out after {timeout} seconds")
        except httpx.RequestError as e:
            raise ServiceCallException(f"Failed to connect to server: {str(e)}")

    async def list_models(self, provider: str | None = None) -> list[dict[str, str]]:
        """
        Lists all available models from the server.
        
        Args:
            provider: Optional provider name to filter models
        
        Returns:
            list[dict[str, str]]: List of models as provider/name pairs.
            Example:
            [
                {"provider": "AZURE_OPENAI", "name": "gpt-4"},
                {"provider": "OPENAI", "name": "gpt-4-mini"},
                {"provider": "AWS", "name": "claude-3-haiku"},
            ]

        Raises:
            ServiceCallException: When there is an error retrieving the model list
        """
        try:
            response = await self._client.get(f"{self.base_url}/list_models")
                
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get("detail", {}).get("message", str(error_data))
                raise ServiceCallException(f"Failed to list models: {error_msg}")
                
            return response.json()
        except httpx.RequestError as e:
            raise ServiceCallException(f"Failed to connect to server: {str(e)}")

    async def list_providers(self) -> list[str]:
        """
        Lists all available providers from the server.

        Returns:
            list[str]: List of provider names

        Raises:
            ServiceCallException: When there is an error retrieving the provider list
        """
        try:
            response = await self._client.get(f"{self.base_url}/list_providers")
                
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get("detail", {}).get("message", str(error_data))
                raise ServiceCallException(f"Failed to list providers: {error_msg}")
                
            return response.json()
        except httpx.RequestError as e:
            raise ServiceCallException(f"Failed to connect to server: {str(e)}")

    def set_model(self, provider: str, name: str):
        """
        Sets the model to be used in subsequent chat requests.
        
        Args:
            provider: Provider name (e.g., "AWS", "AZURE")
            name: Model name (e.g., "claude-3-haiku", "gpt-4")
        """
        self.provider = provider
        self.name = name
        
    async def check_model_credentials(self) -> bool:
        """
        Checks if credentials are set for the current model.
        Should be called after setting a model to verify credentials.
        
        Returns:
            bool: True if credentials are set, False otherwise
            
        Raises:
            ValueError: If model is not set
            CredentialsException: When credentials are not set
        """
        if not self.provider or not self.name:
            raise ValueError("Model is not set, please set it with client.set_model(provider, name) first!")
            
        return await self.check_credentials(raise_exception=True)

    async def check_credentials(self, provider: str = None, name: str = None, raise_exception: bool = False) -> bool:
        """
        Checks if credentials are set for the given provider and model.
        
        Args:
            provider: Provider name (e.g., "AWS", "AZURE"). Uses the currently set provider if None.
            name: Model name (e.g., "claude-3-haiku", "gpt-4"). Uses the currently set model if None.
            raise_exception: Whether to raise an exception if credentials are not set
            
        Returns:
            bool: True if credentials are set, False otherwise
            
        Raises:
            ValueError: When provider or model is not set
            CredentialsException: When credentials are not set and raise_exception is True
        """
        # Use current provider and name if not provided
        provider = provider or self.provider
        name = name or self.name
        
        if not provider or not name:
            raise ValueError("Provider and model name must be set.")
            
        try:
            response = await self._client.get(f"{self.base_url}/check_credentials/{provider}/{name}")
                
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get("detail", {}).get("message", str(error_data))
                
                if raise_exception:
                    raise CredentialsException(f"Credentials check failed: {error_msg}")
                return False
                
            return True
                
        except httpx.RequestError as e:
            if raise_exception:
                raise ServiceCallException(f"Failed to connect to server: {str(e)}")
            return False

    async def chat(self, request: LLMRequest, timeout: float | None = None) -> LLMResponse:
        """
        Sends a chat request to the server.

        Args:
            request: LLMRequest object containing the conversation and parameters
            timeout: Optional timeout override for this specific request

        Returns:
            LLMResponse: Server response containing the model output

        Raises:            
            ValueError: When model is not set
            ModelNotFoundException: When the model is not found on the backend
            CredentialsException: When credentials are not set
            InternalConversionException: When the internal conversion to the particular provider fails
            ServiceCallException: When the service call fails for any reason
            ServiceCallThrottlingException: When the service call is throttled but the number retries is exhausted
            StructuredResponseException: When the structured response parsing fails
            TimeoutException: When the request times out
        """
        if not self.provider or not self.name:
            raise ValueError("Model is not set, please set it with client.set_model(provider, name) first!")
        
        # Set timeout for this specific request if provided
        original_timeout = self._client.timeout
        if timeout is not None:
            validated_timeout = self._validate_timeout(timeout)
            self._client.timeout = httpx.Timeout(validated_timeout)
            
        response_class = request.response_class
        response_format = request.response_format

        request_data = request.model_dump(mode="json")

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/{self.provider}/{self.name}",
                json=request_data
            )
            
            # Handle non-200 responses
            if response.status_code != 200:
                error_data = response.json()
                error_type = error_data.get("detail", {}).get("error", "unknown_error")
                error_msg = error_data.get("detail", {}).get("message", str(error_data))

                if response.status_code == 404 and error_type == "model_not_found":
                    raise ModelNotFoundException(error_msg)
                elif response.status_code == 400 and error_type == "internal_conversion_error":
                    raise InternalConversionException(error_msg)
                elif response.status_code == 429 and error_type == "service_throttling":
                    raise ServiceCallThrottlingException(error_msg)
                elif response.status_code == 422 and error_type == "structured_response_error":
                    raise StructuredResponseException(
                        error_msg,
                        xml=error_data.get("detail", {}).get("xml", ""),
                        return_class=error_data.get("detail", {}).get("return_class")
                    )
                elif response.status_code == 401 and error_type == "credentials_not_set":
                    raise CredentialsException(error_msg)
                elif response.status_code == 502 and error_type == "service_call_error":
                    raise ServiceCallException(error_msg)
                else:
                    raise ServiceCallException(f"Unexpected error: {error_msg}")

            llm_response_as_json = response.json()
            llm_response = LLMResponse.model_validate(llm_response_as_json)

            # Manually convert to StructuredResponse if needed
            if response_format is LLMResponseFormat.XML and response_class is not str:
                llm_response.output = response_class.from_text(llm_response.output)

            return llm_response

        except httpx.TimeoutException as e:
            rprint(f"[bold red]Request timed out after {self._client.timeout.read:.1f} seconds[/bold red]")
            
            current_timeout = self._client.timeout.read if hasattr(self._client.timeout, 'read') else self._client.timeout
            raise TimeoutException(f"Request timed out after {current_timeout:.1f} seconds") from e
        except httpx.RequestError as e:
            rprint(f"[bold red]Request error: {str(e)}[/bold red]")

            if isinstance(e, httpx.ReadTimeout):
                current_timeout = self._client.timeout.read if hasattr(self._client.timeout, 'read') else self._client.timeout
                raise TimeoutException(f"Read timeout after {current_timeout:.1f} seconds") from e
            raise ServiceCallException(f"Failed to connect to server: {str(e)}")
        finally:
            # Restore original timeout
            if timeout is not None:
                self._client.timeout = original_timeout

    async def model_health_check(self, timeout: float = 5.0) -> bool:
        """
        Performs a quick test of the LLM by sending a simple message.
        Should be called after setting a model to verify it's working.
        
        Args:
            timeout: Maximum time to wait for test response in seconds
            
        Returns:
            bool: True if test was successful
            
        Raises:
            ValueError: If model is not set
            ServiceCallException: If the test fails for any reason
            TimeoutException: If the request times out
        """
        if not self.provider or not self.name:
            raise ValueError("Model is not set, please set it with client.set_model(provider, name) first!")

        timeout = self._validate_timeout(timeout)
        
        try:
            # Create a minimal test conversation
            from llm_serv.conversation.conversation import Conversation
            from llm_serv.providers.base import LLMRequest
            
            request = LLMRequest(
                conversation=Conversation.from_prompt("1+1="),
                max_completion_tokens=5,
                temperature=0.0
            )
            
            response = await self.chat(request, timeout=timeout)
            return True
                
        except Exception as e:
            # Preserve the original exception type if it's one we already handle
            if isinstance(e, (ServiceCallException, TimeoutException)):
                raise
            # Wrap other exceptions in ServiceCallException
            raise ServiceCallException(f"Model test failed: {str(e)}") from e
    