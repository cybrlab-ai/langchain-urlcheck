"""Tests for MCP adapter compatibility helper."""

from langchain_urlcheck import get_mcp_server_config


class TestGetMcpServerConfig:
    """Tests for get_mcp_server_config function."""

    def test_returns_valid_config(self) -> None:
        """Should return valid MultiServerMCPClient config."""
        config = get_mcp_server_config(api_key="test-key")

        assert config["transport"] == "http"
        assert config["url"] == "https://urlcheck.ai/mcp"
        assert config["headers"]["X-API-Key"] == "test-key"
        assert "MCP-Protocol-Version" in config["headers"]

    def test_custom_url(self) -> None:
        """Should accept custom MCP URL."""
        config = get_mcp_server_config(
            api_key="test-key",
            mcp_url="https://custom.example.com/mcp",
        )

        assert config["url"] == "https://custom.example.com/mcp"

    def test_custom_timeout(self) -> None:
        """Should accept custom timeout."""
        config = get_mcp_server_config(
            api_key="test-key",
            timeout_seconds=60,
        )

        assert config["timeout"] == 60

    def test_empty_api_key_omits_header(self) -> None:
        """Should omit X-API-Key header when api_key is empty (free tier)."""
        config = get_mcp_server_config(api_key="")
        assert "X-API-Key" not in config["headers"]
        assert "MCP-Protocol-Version" in config["headers"]

    def test_no_api_key_defaults_to_free_tier(self) -> None:
        """Should work without api_key argument (free tier)."""
        config = get_mcp_server_config()
        assert "X-API-Key" not in config["headers"]
        assert config["url"] == "https://urlcheck.ai/mcp"
