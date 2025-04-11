from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelProvider(BaseModel):
    name: str
    config: dict = {}


class Model(BaseModel):
    id: str  # format: provider/model_name
    internal_model_id: str
    name: str
    provider: ModelProvider    
    
    max_tokens: int
    max_output_tokens: int
    capabilities: dict = {}
    config: dict = {}

    # TODO implement __str__ and __repr__
    
    @property
    def image_support(self) -> bool:
        return self.capabilities.get("image_support", False)
        
    @property
    def document_support(self) -> bool:
        return self.capabilities.get("document_support", False)


class LLMService:
    _instance = None
    _initialized = False
    providers: list[ModelProvider] = []
    models: list[Model] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance

    def __init__(self, yaml_path: Path | None = None):
        if not LLMService._initialized:
            self._initialize(yaml_path)
            LLMService._initialized = True

    def _initialize(self, yaml_path: Path | str | None = None):
        # Get the path to models.yaml
        if yaml_path is None:
            yaml_path = Path(__file__).parent / "models.yaml"
        else:
            if isinstance(yaml_path, str):
                yaml_path = Path(yaml_path)
            if not yaml_path.exists():
                raise FileNotFoundError(f"Models file not found at '{yaml_path}'!")

        # Load the models.yaml file
        with open(yaml_path, "r") as file:
            data: dict = yaml.safe_load(file)

        # Initialize the models and providers
        models = []
        models_data: dict = data.get("MODELS", {})
        providers_data: dict = data.get("PROVIDERS", {})

        for model_id, model_data in models_data.items():
            provider_name, model_name = model_id.split("/")
            
            # Create or get the provider
            provider = None
            for existing_provider in self.providers:
                if existing_provider.name == provider_name:
                    provider = existing_provider
                    break
                    
            if provider is None:
                if provider_name not in providers_data:
                    raise ValueError(f"Provider '{provider_name}' referenced in model '{model_id}' but not defined in PROVIDERS section")
                provider = ModelProvider(name=provider_name, config=providers_data[provider_name].get("config", {}))
                self.providers.append(provider)
            
            # Create the model
            model = Model(
                provider=provider,
                name=model_name,
                id=model_id,
                internal_model_id=model_data["internal_model_id"],
                max_tokens=model_data["max_tokens"],
                max_output_tokens=model_data["max_output_tokens"],
                capabilities=model_data.get("capabilities", {}),
                config=model_data.get("config", {}),
            )
            models.append(model)

        self.models = models

    @staticmethod
    def get_model(model_id: str) -> Model:
        """
        Get a model by its ID (format: provider/model_name) or name.
        
        Args:
            model_id: The model ID in the format "provider/model_name" or just the model name
            
        Returns:
            Model: The model object
            
        Raises:
            ValueError: If no model is found
        """
        service = LLMService()
        
        if not service.models:
            service._initialize()

        if "/" in model_id:
            provider_name, model_name = model_id.split("/")
            for model in service.models:
                if model.provider.name.upper() == provider_name.upper() and model.name == model_name:
                    return model
        else:
            # Try to find by name only
            for model in service.models:
                if model.name == model_id:
                    return model

        raise ValueError(f"No model found for ID '{model_id}'")

    @staticmethod
    def _check_model_id(model_id: str) -> bool:
        """
        Check if the model ID is valid.
        """
        try:
            LLMService.get_model(model_id)
            return True
        except ValueError:
            return False

    @staticmethod
    def list_providers() -> list[ModelProvider]:
        """
        List all available providers.
        """
        service = LLMService()
        return service.providers

    @staticmethod
    def list_models(provider: str | None = None) -> list[Model]:
        """
        List all available models.
        
        Args:
            provider: Optional provider name to filter models
            
        Returns:
            list[Model]: List of models
        """
        service = LLMService()
        
        if provider is None:
            return service.models  
        else:
            return [model for model in service.models if model.provider.name.upper() == provider.upper()]

    @staticmethod
    def get_provider(model: Model | str):
        """
        Factory function to create an LLM service instance based on the provider.

        Args:
            model: Model configuration from the registry or a string with the format "provider/model"

        Returns:
            LLMProvider: An instance of the appropriate LLM provider

        Raises:
            ValueError: If the provider is not supported
        """
        if isinstance(model, str):
            model = LLMService.get_model(model)
            
        provider_name = model.provider.name.upper()

        match provider_name:
            case "AWS":
                # Check credentials
                from llm_serv.core.providers.aws import check_credentials
                check_credentials()

                # Create LLM service
                from llm_serv.core.providers.aws import AWSLLMProvider
                return AWSLLMProvider(model)
            
            case "AZURE":
                # Check credentials
                from llm_serv.core.providers.azure import check_credentials
                check_credentials()

                # Create LLM service
                from llm_serv.core.providers.azure import AzureOpenAILLMProvider
                return AzureOpenAILLMProvider(model)
            
            case "OPENAI":
                # Check credentials
                from llm_serv.core.providers.oai import check_credentials
                check_credentials()

                # Create LLM service
                from llm_serv.core.providers.oai import OpenAILLMProvider
                return OpenAILLMProvider(model)
            
            case _:
                raise ValueError(f"Unsupported provider: {provider_name}.")

# Initialize the service at module load time
LLMService()


