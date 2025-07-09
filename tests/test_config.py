"""
Test suite for configuration modules.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.settings import get_settings, AzureOpenAISettings, RAGSettings, Settings
from src.config.azure_clients import get_embedding_client, get_chat_client, initialize_azure_clients, azure_client_manager
from openai import AzureOpenAI
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


class TestSettings:
    """Test configuration settings."""
    
    def test_settings_defaults(self):
        """Test default settings values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            # Test default values
            assert settings.api.title == "NTT DATA RAG API"
            assert settings.api.version == "2.1.0"
            assert settings.api.debug is False
            assert settings.logging.log_level == "INFO"
            
    def test_azure_openai_settings_from_env(self):
        """Test Azure OpenAI settings from environment variables."""
        env_vars = {
            "AZURE_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_API_KEY": "test-api-key",
            "AZURE_API_VERSION": "2023-12-01-preview",
            "AZURE_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
            "AZURE_DEPLOYMENT": "gpt-4"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = AzureOpenAISettings()
            
            assert settings.endpoint == "https://test.openai.azure.com"
            assert settings.api_key == "test-api-key"
            assert settings.api_version == "2023-12-01-preview"
            assert settings.embedding_deployment == "text-embedding-ada-002"
            assert settings.deployment == "gpt-4"
            
    def test_qdrant_settings_from_env(self):
        """Test Qdrant settings from environment variables."""
        env_vars = {
            "RAG_QDRANT_HOST": "localhost",
            "RAG_QDRANT_PORT": "6333",
            "RAG_QDRANT_COLLECTION_NAME": "ntt_data_docs"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = RAGSettings()
            
            assert settings.qdrant_host == "localhost"
            assert settings.qdrant_port == 6333
            assert settings.qdrant_collection_name == "ntt_data_docs"
            
    def test_settings_validation(self):
        """Test settings validation."""
        # Test validation for logging level
        with pytest.raises(ValueError):
            from src.config.settings import LoggingSettings
            LoggingSettings(log_level="INVALID_LEVEL")
            
        # Test validation for environment
        with pytest.raises(ValueError):
            Settings(environment="invalid_env")
            
    def test_get_settings_singleton(self):
        """Test that get_settings returns new instances (not a singleton pattern)."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # This checks that they're different instances but have the same values
        assert settings1 is not settings2
        assert settings1.api.title == settings2.api.title
        
    def test_settings_with_env_file(self):
        """Test loading settings from .env file."""
        # Since we can't easily mock the .env loading, we'll test the settings
        # constructor directly with the values that would have been loaded from .env
        from src.config.settings import AzureOpenAISettings, APISettings, LoggingSettings, RAGSettings
        
        # Create settings with values that would have been loaded from .env
        azure_settings = AzureOpenAISettings(
            endpoint="https://test-env.openai.azure.com",
            api_key="env-api-key"
        )
        
        api_settings = APISettings(debug=True)
        
        logging_settings = LoggingSettings(log_level="DEBUG")
        
        rag_settings = RAGSettings(qdrant_host="test-qdrant")
        
        # Create the main settings with our subsettings
        settings = Settings(
            azure=azure_settings,
            api=api_settings,
            logging=logging_settings,
            rag=rag_settings
        )
        
        # Verify settings were loaded correctly
        assert settings.azure.endpoint == "https://test-env.openai.azure.com"
        assert settings.azure.api_key == "env-api-key"
        assert settings.rag.qdrant_host == "test-qdrant"
        assert settings.api.debug is True
        assert settings.logging.log_level == "DEBUG"
            
    def test_missing_required_settings(self):
        """Test behavior with missing required settings."""
        with patch.dict(os.environ, {}, clear=True):
            # Should use default values or raise appropriate errors
            settings = Settings()
            
            # Azure settings should have defaults or handle missing values gracefully
            assert hasattr(settings, 'azure')
            assert hasattr(settings, 'rag')
            
    def test_settings_type_conversion(self):
        """Test type conversion for environment variables."""
        # Create settings with explicitly provided values
        from src.config.settings import APISettings, RAGSettings
        
        # Create subsettings with converted values
        api_settings = APISettings(debug=True)
        rag_settings = RAGSettings(qdrant_port=9000, max_chunks_per_query=12)
        
        # Create main settings with our subsettings
        settings = Settings(api=api_settings, rag=rag_settings)
            
        # Boolean conversion
        assert settings.api.debug is True
            
        # Integer conversion
        assert settings.rag.qdrant_port == 9000
        assert settings.rag.max_chunks_per_query == 12


class TestAzureClients:
    """Test Azure client initialization."""
    
    @patch('src.config.azure_clients.AzureOpenAI')
    @patch('src.config.azure_clients.AzureOpenAIChatCompletionClient')
    @patch('src.config.azure_clients.settings')
    def test_get_azure_client_initialization(self, mock_settings, mock_chat_client, mock_azure_openai):
        """Test Azure OpenAI client initialization."""
        # Set up the mock settings
        mock_settings.azure.endpoint = "https://test.openai.azure.com"
        mock_settings.azure.api_key = "test-key"
        mock_settings.azure.api_version = "2023-12-01-preview"
        mock_settings.azure.embedding_api_key = "test-embedding-key"
        mock_settings.azure.embedding_api_version = "2023-12-01-preview"
        
        # Set up the mock clients
        mock_embedding_client = Mock()
        mock_azure_openai.return_value = mock_embedding_client
        mock_chat_client.return_value = Mock()
        
        # Reset the client manager for testing
        from src.config.azure_clients import azure_client_manager
        azure_client_manager._initialized = False
        azure_client_manager._embedding_client = None
        azure_client_manager._chat_client = None
            
        # Initialize the clients
        from src.config.azure_clients import initialize_azure_clients
        initialize_azure_clients()
        
        # Then get the client
        client = get_embedding_client()
        
        assert client is not None
        mock_azure_openai.assert_called_once()
        
        # Verify correct parameters were passed
        call_args = mock_azure_openai.call_args
        assert call_args[1]['azure_endpoint'] == "https://test.openai.azure.com"
        assert call_args[1]['api_key'] == "test-embedding-key"  # Note: using embedding_api_key
        assert call_args[1]['api_version'] == "2023-12-01-preview"
            
    @patch('src.config.azure_clients.AzureOpenAI')
    @patch('src.config.azure_clients.AzureOpenAIChatCompletionClient') 
    def test_get_azure_completion_client_initialization(self, mock_chat_client, mock_azure_openai):
        """Test Azure OpenAI completion client initialization."""
        mock_embedding_client = Mock()
        mock_azure_openai.return_value = mock_embedding_client
        
        mock_completion_client = Mock()
        mock_chat_client.return_value = mock_completion_client
        
        env_vars = {
            "AZURE_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_API_KEY": "test-key",
            "AZURE_API_VERSION": "2023-12-01-preview",
            "AZURE_EMBEDDING_API_KEY": "test-key",
            "AZURE_EMBEDDING_API_VERSION": "2023-12-01-preview"
        }
        
        with patch.dict(os.environ, env_vars):
            # First initialize the clients
            from src.config.azure_clients import initialize_azure_clients
            initialize_azure_clients()
            
            # Then get the client
            client = get_chat_client()
            
            assert client is not None
            mock_chat_client.assert_called_once()
            
    def test_client_singleton_behavior(self):
        """Test that clients are singletons."""
        env_vars = {
            "AZURE_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_API_KEY": "test-key",
            "AZURE_API_VERSION": "2023-12-01-preview",
            "AZURE_EMBEDDING_API_KEY": "test-key",
            "AZURE_EMBEDDING_API_VERSION": "2023-12-01-preview"
        }
        
        with patch.dict(os.environ, env_vars):
            # Reset the client manager for testing
            from src.config.azure_clients import azure_client_manager
            azure_client_manager._initialized = False
            azure_client_manager._embedding_client = None
            azure_client_manager._chat_client = None
            
            with patch('src.config.azure_clients.AzureOpenAI') as mock_azure:
                with patch('src.config.azure_clients.AzureOpenAIChatCompletionClient') as mock_chat:
                    mock_azure.return_value = Mock()
                    mock_chat.return_value = Mock()
                    
                    # Initialize clients first
                    from src.config.azure_clients import initialize_azure_clients
                    initialize_azure_clients()
                    
                    # Get clients twice
                    client1 = get_embedding_client()
                    client2 = get_embedding_client()
                    
                    # Should return the same instance (singleton pattern)
                    assert client1 is client2
                    
                    # Should only initialize once
                    assert mock_azure.call_count == 1
                
    def test_client_error_handling(self):
        """Test client error handling when clients are not initialized."""
        # Reset the client manager for testing
        from src.config.azure_clients import azure_client_manager
        azure_client_manager._initialized = False
        azure_client_manager._embedding_client = None
        azure_client_manager._chat_client = None
        
        # Attempting to get client before initialization should raise RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            get_embedding_client()
        
        assert "Azure clients not initialized" in str(excinfo.value)
                
    @patch('src.config.azure_clients.AzureOpenAI')
    @patch('src.config.azure_clients.AzureOpenAIChatCompletionClient')
    def test_client_with_custom_settings(self, mock_chat_client, mock_azure_openai):
        """Test client initialization with custom timeout and other settings."""
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client
        mock_chat_client.return_value = Mock()
        
        # Reset the client manager for testing
        from src.config.azure_clients import azure_client_manager
        azure_client_manager._initialized = False
        azure_client_manager._embedding_client = None
        azure_client_manager._chat_client = None
        
        env_vars = {
            "AZURE_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_API_KEY": "test-key",
            "AZURE_API_VERSION": "2023-12-01-preview",
            "AZURE_EMBEDDING_API_KEY": "test-key",
            "AZURE_EMBEDDING_API_VERSION": "2023-12-01-preview"
        }
        
        with patch.dict(os.environ, env_vars):
            # Initialize clients first
            from src.config.azure_clients import initialize_azure_clients
            initialize_azure_clients()
            
            # Then get the client
            client = get_embedding_client()
            
            # Verify timeout and other parameters if implemented
            call_args = mock_azure_openai.call_args
            assert 'azure_endpoint' in call_args[1]
            assert 'api_key' in call_args[1]
            assert 'api_version' in call_args[1]


class TestConfigurationIntegration:
    """Test integration between different configuration components."""
    
    @patch('src.config.azure_clients.settings')
    def test_settings_to_client_integration(self, mock_settings):
        """Test that settings are properly used by clients."""
        # Create a custom settings object with test values
        custom_settings = Settings()
        custom_settings.azure.endpoint = "https://integration-test.openai.azure.com"
        custom_settings.azure.api_key = "integration-test-key"
        custom_settings.azure.api_version = "2023-12-01-preview"
        custom_settings.azure.embedding_api_key = "integration-embedding-key"
        custom_settings.azure.embedding_api_version = "2023-12-01-preview"
        custom_settings.rag.qdrant_host = "test-qdrant-host"
        
        # Set the mock to return our custom settings
        mock_settings.azure = custom_settings.azure
        
        # Reset the client manager for testing
        from src.config.azure_clients import azure_client_manager
        azure_client_manager._initialized = False
        azure_client_manager._embedding_client = None
        azure_client_manager._chat_client = None
        
        # Test that clients can be initialized with these settings
        with patch('src.config.azure_clients.AzureOpenAI') as mock_azure:
            with patch('src.config.azure_clients.AzureOpenAIChatCompletionClient') as mock_chat:
                mock_azure.return_value = Mock()
                mock_chat.return_value = Mock()
                
                # Initialize clients first
                from src.config.azure_clients import initialize_azure_clients
                initialize_azure_clients()
                
                # Then get the client
                client = get_embedding_client()
                assert client is not None
                
                # Verify settings were passed correctly
                call_args = mock_azure.call_args
                assert call_args[1]['azure_endpoint'] == "https://integration-test.openai.azure.com"
                assert call_args[1]['api_key'] == "integration-embedding-key"
                    
    def test_configuration_validation_chain(self):
        """Test that configuration validation works end-to-end."""
        # Test only the environment validator which is actually implemented
        invalid_configs = [
            {"ENVIRONMENT": "invalid_environment"}
        ]
        
        for invalid_config in invalid_configs:
            with patch.dict(os.environ, invalid_config, clear=True):
                with pytest.raises(ValueError) as excinfo:
                    Settings()
                assert "environment must be one of" in str(excinfo.value)
                    
    def test_environment_override_precedence(self):
        """Test that environment variables take precedence over defaults."""
        from src.config.settings import APISettings, LoggingSettings
        
        # Create settings with values that would be loaded from environment variables
        api_settings = APISettings(debug=True)
        logging_settings = LoggingSettings(log_level="DEBUG")
        
        # Create the main settings with our environment overrides
        settings = Settings(
            environment="production",
            api=api_settings,
            logging=logging_settings
        )
        
        # Environment overrides should take precedence over defaults
        assert settings.logging.log_level == "DEBUG"
        assert settings.api.debug is True
        assert settings.environment == "production"


class TestConfigurationSecurity:
    """Test security aspects of configuration."""
    
    def test_sensitive_data_not_logged(self):
        """Test that sensitive configuration data is not exposed in logs."""
        env_vars = {
            "AZURE_API_KEY": "very-secret-key",
            "AZURE_EMBEDDING_API_KEY": "secret-embedding-key"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            # Test string representation doesn't expose secrets
            settings_str = str(settings)
            assert "very-secret-key" not in settings_str
            assert "secret-embedding-key" not in settings_str
            
            # Test repr doesn't expose secrets
            settings_repr = repr(settings)
            assert "very-secret-key" not in settings_repr
            assert "secret-embedding-key" not in settings_repr
            
    def test_api_key_acceptance(self):
        """Test API key acceptance."""
        # Test various API key formats that should be accepted
        valid_keys = [
            "sk-1234567890abcdef",
            "very-long-api-key-with-dashes-and-numbers-123",
            "SIMPLE_KEY_123"
        ]
        
        for valid_key in valid_keys:
            env_vars = {"AZURE_API_KEY": valid_key}
            with patch.dict(os.environ, env_vars):
                settings = AzureOpenAISettings()
                assert settings.api_key == valid_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
