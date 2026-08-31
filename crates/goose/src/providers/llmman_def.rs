use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;
use url::Url;

use crate::providers::base::ProviderDef;
use goose_providers::{
    api_client::{ApiClient, AuthMethod},
    base::{ConfigKey, ProviderMetadata},
    ollama::{OllamaProvider, OllamaProviderBuilder, OLLAMA_TIMEOUT},
};

/// llmman (https://github.com/llmmanorg/llmman) serves the Ollama API, so it
/// reuses that provider and changes only the name, the config key and the
/// default port it listens on.
pub const LLMMAN_PROVIDER_NAME: &str = "llmman";
pub const LLMMAN_HOST: &str = "localhost";
pub const LLMMAN_DEFAULT_PORT: u16 = 17434;
pub const LLMMAN_DEFAULT_MODEL: &str = "qwen3.8";
pub const LLMMAN_KNOWN_MODELS: &[&str] = &[LLMMAN_DEFAULT_MODEL, "gemma4"];
pub const LLMMAN_DOC_URL: &str = "https://github.com/llmmanorg/llmman";

pub struct LlmmanProviderDef;

impl goose_providers::base::ProviderDescriptor for LlmmanProviderDef {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            LLMMAN_PROVIDER_NAME,
            "llmman",
            "Local models distributed as OCI artifacts",
            LLMMAN_DEFAULT_MODEL,
            LLMMAN_KNOWN_MODELS.to_vec(),
            LLMMAN_DOC_URL,
            vec![
                ConfigKey::new("LLMMAN_HOST", true, false, Some(LLMMAN_HOST), true),
                ConfigKey::new(
                    "LLMMAN_TIMEOUT",
                    false,
                    false,
                    Some(&(OLLAMA_TIMEOUT.to_string())),
                    false,
                ),
            ],
        )
        .with_setup(
            crate::providers::catalog::ProviderSetupMetadata::new(
                crate::providers::catalog::ProviderSetupCategory::Model,
                crate::providers::catalog::ProviderSetupMethod::ConfigFields,
                crate::providers::catalog::ProviderSetupGroup::Default,
            )
            .with_docs_url(LLMMAN_DOC_URL)
            .with_field(
                "LLMMAN_HOST",
                "Host",
                Some("localhost or http://localhost:17434"),
                Some("http://localhost:17434"),
            ),
        )
    }
}

impl ProviderDef for LlmmanProviderDef {
    type Provider = OllamaProvider;

    fn from_env(
        _extensions: Vec<crate::config::ExtensionConfig>,
        tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> BoxFuture<'static, Result<Self::Provider>> {
        Box::pin(from_env(tls_config))
    }
}

pub async fn from_env(
    tls_config: Option<crate::providers::api_client::TlsConfig>,
) -> Result<OllamaProvider> {
    let config = crate::config::Config::global();
    let host: String = config
        .get_param("LLMMAN_HOST")
        .unwrap_or_else(|_| LLMMAN_HOST.to_string());

    let timeout: Duration =
        Duration::from_secs(config.get_param("LLMMAN_TIMEOUT").unwrap_or(OLLAMA_TIMEOUT));

    let base = if host.starts_with("http://") || host.starts_with("https://") {
        host.clone()
    } else {
        format!("http://{}", host)
    };

    let mut base_url = Url::parse(&base).map_err(|e| anyhow::anyhow!("Invalid base URL: {e}"))?;

    let explicit_port = host.contains(':');
    let is_localhost = host == "localhost" || host == "127.0.0.1" || host == "::1";

    if base_url.port().is_none() && !explicit_port && !host.starts_with("http") && is_localhost {
        base_url
            .set_port(Some(LLMMAN_DEFAULT_PORT))
            .map_err(|_| anyhow::anyhow!("Failed to set default port"))?;
    }

    let api_client = ApiClient::with_timeout_and_tls(
        base_url.to_string(),
        AuthMethod::NoAuth,
        timeout,
        tls_config,
    )?
    .with_request_builder(crate::session_context::session_id_request_builder());

    Ok(OllamaProviderBuilder::new(api_client)
        .name(LLMMAN_PROVIDER_NAME)
        .options(crate::providers::ollama_def::options_from_config())
        .build())
}
