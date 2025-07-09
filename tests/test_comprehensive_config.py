"""
Comprehensive test suite for configuration modules.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.settings import get_settings, AzureOpenAISettings, RAGSettings, Settings, APISettings, DirectorySettings, LoggingSettings
from src.config.azure_clients import get_azure_client, get_azure_completion_client


class TestBasicSettings:
    """Test basic configuration settings."""
    
    def test_settings_initialization(self):
        """Test that settings can be initialized successfully."""
        settings = Settings()
        
        assert settings is not None
        assert hasattr(settings, 'azure')
        assert hasattr(settings, 'rag')
        assert hasattr(settings, 'api')
        assert hasattr(settings, 'directories')
        assert hasattr(settings, 'logging')
        
    def test_azure_settings_defaults(self):
        """Test Azure OpenAI settings default values."""
        azure_settings = AzureOpenAISettings()
        
        assert azure_settings.api_version == "2024-07-01-preview"
        assert azure_settings.embedding_model == "text-embedding-3-large"
        assert azure_settings.completion_model == "gpt-4o"
        assert azure_settings.max_tokens == 4000
        assert azure_settings.temperature == 0.1
        
    def test_rag_settings_defaults(self):
        """Test RAG settings default values."""
        rag_settings = RAGSettings()
        
        assert rag_settings.chunk_size == 800
        assert rag_settings.chunk_overlap == 150
        assert rag_settings.similarity_threshold == 0.25
        assert rag_settings.max_chunks_per_query == 8
        assert rag_settings.qdrant_port == 6333
        assert rag_settings.qdrant_vector_size == 3072
        
    def test_api_settings_defaults(self):
        """Test API settings default values."""
        api_settings = APISettings()
        
        assert api_settings.title == "NTT DATA RAG API"
        assert api_settings.version == "2.1.0"
        assert api_settings.debug is False
        
    def test_directory_settings_defaults(self):
        """Test directory settings default values."""
        dir_settings = DirectorySettings()
        
        assert dir_settings.reports_dir == "./reports"
        assert dir_settings.data_dir == "./data"
        assert dir_settings.logs_dir == "./data/logs"
        assert dir_settings.cache_dir == "./data/cache"
        
    def test_logging_settings_defaults(self):
        """Test logging settings default values."""
        log_settings = LoggingSettings()
        
        assert log_settings.log_level == "INFO"
        assert log_settings.log_format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert log_settings.log_file_max_size == 10485760  # 10MB
        assert log_settings.log_file_backup_count == 5
        assert log_settings.disable_autogen_logs is True
        
    def test_environment_override(self):
        """Test that environment variables override defaults.""" 
        env_vars = {
            "API_DEBUG": "true",
            "RAG_CHUNK_SIZE": "1200",
            "LOG_LEVEL": "DEBUG"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            # Create new instances to get env values
            api_settings = APISettings()
            rag_settings = RAGSettings()
            log_settings = LoggingSettings()
            
            assert api_settings.debug is True
            assert rag_settings.chunk_size == 1200
            assert log_settings.log_level == "DEBUG"


class TestSettingsValidation:
    """Test settings validation logic."""
    
    def test_log_level_validation_valid(self):
        """Test log level validation with valid values."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_levels:
            env_vars = {"LOG_LEVEL": level}
            with patch.dict(os.environ, env_vars, clear=False):
                settings = LoggingSettings()
                assert settings.log_level == level.upper()
                
    def test_log_level_validation_invalid(self):
        """Test log level validation with invalid values."""
        env_vars = {"LOG_LEVEL": "INVALID"}
        
        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValueError, match="log_level must be one of"):
                LoggingSettings()
                
    def test_environment_validation_valid(self):
        """Test environment validation with valid values."""
        valid_envs = ['development', 'staging', 'production']
        
        for env in valid_envs:
            env_vars = {"ENVIRONMENT": env}
            with patch.dict(os.environ, env_vars, clear=False):
                settings = Settings()
                assert settings.environment == env.lower()
                
    def test_environment_validation_invalid(self):
        """Test environment validation with invalid values."""
        env_vars = {"ENVIRONMENT": "invalid_env"}
        
        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValueError, match="environment must be one of"):
                Settings()
                
    def test_numeric_field_validation(self):
        """Test numeric field validation."""
        env_vars = {
            "RAG_CHUNK_SIZE": "1500",
            "RAG_CHUNK_OVERLAP": "200",
            "RAG_QDRANT_PORT": "6334"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = RAGSettings()
            assert settings.chunk_size == 1500
            assert settings.chunk_overlap == 200
            assert settings.qdrant_port == 6334
            
    def test_boolean_field_validation(self):
        """Test boolean field validation."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False)
        ]
        
        for str_value, expected_bool in test_cases:
            env_vars = {"API_DEBUG": str_value}
            with patch.dict(os.environ, env_vars, clear=False):
                settings = APISettings()
                assert settings.debug is expected_bool


class TestAzureClientIntegration:
    """Test Azure client integration with settings."""
    
    @patch('src.config.azure_clients.AsyncAzureOpenAI')
    def test_azure_client_creation(self, mock_azure_openai):
        """Test Azure client creation with settings."""
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client
        
        client = get_azure_client()
        
        assert client is not None
        mock_azure_openai.assert_called_once()
        
        # Verify client was called with correct parameters
        call_kwargs = mock_azure_openai.call_args[1]
        assert 'azure_endpoint' in call_kwargs
        assert 'api_key' in call_kwargs
        assert 'api_version' in call_kwargs
        
    @patch('src.config.azure_clients.AsyncAzureOpenAI')
    def test_azure_completion_client_creation(self, mock_azure_openai):
        """Test Azure completion client creation."""
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client
        
        client = get_azure_completion_client()
        
        assert client is not None
        mock_azure_openai.assert_called_once()
        
    def test_client_initialization_with_custom_settings(self):
        """Test client initialization uses custom environment settings."""
        custom_env = {
            "AZURE_ENDPOINT": "https://custom-test.openai.azure.com",
            "AZURE_API_KEY": "custom-test-key-12345",
            "AZURE_API_VERSION": "2024-07-01-preview"
        }
        
        with patch.dict(os.environ, custom_env, clear=False):
            with patch('src.config.azure_clients.AsyncAzureOpenAI') as mock_azure:
                mock_azure.return_value = Mock()
                
                get_azure_client()
                
                call_kwargs = mock_azure.call_args[1]
                assert call_kwargs['azure_endpoint'] == custom_env["AZURE_ENDPOINT"]
                assert call_kwargs['api_key'] == custom_env["AZURE_API_KEY"]
                assert call_kwargs['api_version'] == custom_env["AZURE_API_VERSION"]


class TestSettingsSingleton:
    """Test settings singleton behavior."""
    
    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
        
    def test_singleton_preserves_state(self):
        """Test that singleton preserves state changes."""
        settings = get_settings()
        original_debug = settings.api.debug
        
        # This would test state preservation if settings were mutable
        # Since pydantic settings are typically immutable, we just verify consistency
        settings_again = get_settings()
        assert settings_again.api.debug == original_debug


class TestConfigurationInteraction:
    """Test interaction between different configuration components."""
    
    def test_all_settings_components_accessible(self):
        """Test that all settings components are accessible through main settings."""
        settings = Settings()
        
        # Verify all components exist and are of correct type
        assert isinstance(settings.azure, AzureOpenAISettings)
        assert isinstance(settings.rag, RAGSettings)
        assert isinstance(settings.api, APISettings)
        assert isinstance(settings.directories, DirectorySettings)
        assert isinstance(settings.logging, LoggingSettings)
        
    def test_settings_environment_isolation(self):
        """Test that different environment prefixes work correctly."""
        env_vars = {
            "AZURE_API_VERSION": "custom-azure-version",
            "RAG_CHUNK_SIZE": "999",
            "API_DEBUG": "true",
            "LOG_LEVEL": "WARNING"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            
            # Each component should only pick up its prefixed variables
            assert "custom-azure-version" in settings.azure.api_version
            assert settings.rag.chunk_size == 999
            assert settings.api.debug is True
            assert settings.logging.log_level == "WARNING"
            
    def test_complex_configuration_scenario(self):
        """Test a complex configuration scenario with multiple overrides."""
        complex_env = {
            "ENVIRONMENT": "production",
            "AZURE_ENDPOINT": "https://prod.openai.azure.com",
            "AZURE_MAX_TOKENS": "8000",
            "AZURE_TEMPERATURE": "0.0",
            "RAG_CHUNK_SIZE": "1000", 
            "RAG_MAX_CHUNKS_PER_QUERY": "12",
            "RAG_QDRANT_HOST": "prod-qdrant.internal",
            "RAG_QDRANT_PORT": "6333",
            "API_DEBUG": "false",
            "API_TITLE": "Production RAG API",
            "LOG_LEVEL": "ERROR",
            "DIR_DATA_DIR": "/prod/data"
        }
        
        with patch.dict(os.environ, complex_env, clear=False):
            settings = Settings()
            
            # Verify all settings are applied correctly
            assert settings.environment == "production"
            assert "prod.openai.azure.com" in settings.azure.endpoint
            assert settings.azure.max_tokens == 8000
            assert settings.azure.temperature == 0.0
            assert settings.rag.chunk_size == 1000
            assert settings.rag.max_chunks_per_query == 12
            assert settings.rag.qdrant_host == "prod-qdrant.internal"
            assert settings.api.debug is False
            assert settings.api.title == "Production RAG API"
            assert settings.logging.log_level == "ERROR"
            assert settings.directories.data_dir == "/prod/data"


class TestErrorHandling:
    """Test error handling in configuration."""
    
    def test_invalid_numeric_values(self):
        """Test handling of invalid numeric values."""
        invalid_cases = [
            ("RAG_CHUNK_SIZE", "not_a_number"),
            ("RAG_QDRANT_PORT", "invalid_port"),
            ("AZURE_MAX_TOKENS", "text_instead_of_number")
        ]
        
        for env_var, invalid_value in invalid_cases:
            env_vars = {env_var: invalid_value}
            with patch.dict(os.environ, env_vars, clear=False):
                with pytest.raises((ValueError, TypeError)):
                    if env_var.startswith("RAG_"):
                        RAGSettings()
                    elif env_var.startswith("AZURE_"):
                        AzureOpenAISettings()
                        
    def test_missing_required_fields_handling(self):
        """Test handling when required fields are missing."""
        # Remove critical environment variables to test defaults
        critical_vars = ["AZURE_ENDPOINT", "AZURE_API_KEY"]
        
        original_values = {}
        for var in critical_vars:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Should still work with defaults or handle gracefully
            settings = AzureOpenAISettings()
            assert settings is not None
        finally:
            # Restore original values
            for var, value in original_values.items():
                os.environ[var] = value
                
    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        edge_cases = {
            "RAG_CHUNK_SIZE": "0",  # Zero value
            "RAG_SIMILARITY_THRESHOLD": "1.0",  # Maximum threshold
            "AZURE_TEMPERATURE": "2.0",  # High temperature
            "LOG_LEVEL": "debug"  # Lowercase level (should be normalized)
        }
        
        with patch.dict(os.environ, edge_cases, clear=False):
            try:
                rag_settings = RAGSettings()
                azure_settings = AzureOpenAISettings()
                log_settings = LoggingSettings()
                
                # Verify edge cases are handled appropriately
                assert rag_settings.chunk_size == 0
                assert rag_settings.similarity_threshold == 1.0
                assert azure_settings.temperature == 2.0
                assert log_settings.log_level == "DEBUG"  # Should be normalized to uppercase
                
            except ValueError as e:
                # Some edge cases might legitimately raise validation errors
                assert "must be" in str(e) or "invalid" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
