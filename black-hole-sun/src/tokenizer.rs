use crate::{DarkToken, LogitEntry};

const DEFAULT_REPO: &str = "Qwen/Qwen3.5-0.8B";
const DEFAULT_REVISION: &str = "main";
const DEFAULT_FILE: &str = "tokenizer.json";

/// Shared tokenizer loader and decoding helper for black-hole workspace tests.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Initialize the default workspace tokenizer.
    pub fn init() -> Self {
        Self::try_init().unwrap_or_else(|error| panic!("failed to initialize tokenizer: {error}"))
    }

    /// Try to initialize the default workspace tokenizer.
    pub fn try_init() -> Result<Self, String> {
        TokenizerBuilder::new().build()
    }

    /// Create a configurable tokenizer builder.
    pub fn builder() -> TokenizerBuilder {
        TokenizerBuilder::new()
    }

    /// Decode DarkToken predictions into text.
    pub fn decode(&self, tokens: &[DarkToken]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|token| token.predicted).collect();
        self.inner
            .decode(&ids, true)
            .unwrap_or_else(|_| ids.iter().map(|id| id.to_string()).collect())
    }

    /// Encode text into tokenizer ids.
    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoded = self
            .inner
            .encode(text, false)
            .map_err(|error| format!("failed to tokenize input: {error}"))?;

        Ok(encoded.get_ids().iter().map(|&id| id as u32).collect())
    }

    /// Encode text into `DarkToken`s suitable for dark inference requests.
    pub fn darken(&self, text: &str) -> Result<Vec<DarkToken>, String> {
        let token_ids = self.encode_ids(text)?;
        Ok(token_ids
            .into_iter()
            .map(|token_id| DarkToken {
                predicted: token_id,
                dark_knowledge: vec![LogitEntry {
                    token_id,
                    log_prob: 0.0,
                }],
            })
            .collect())
    }

    /// Access the underlying tokenizer for advanced operations.
    pub fn as_inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

#[derive(Clone, Debug)]
pub struct TokenizerBuilder {
    repo: String,
    revision: String,
    file: String,
}

impl Default for TokenizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerBuilder {
    pub fn new() -> Self {
        Self {
            repo: DEFAULT_REPO.to_string(),
            revision: DEFAULT_REVISION.to_string(),
            file: DEFAULT_FILE.to_string(),
        }
    }

    pub fn repo(mut self, repo: impl Into<String>) -> Self {
        self.repo = repo.into();
        self
    }

    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    pub fn file(mut self, file: impl Into<String>) -> Self {
        self.file = file.into();
        self
    }

    pub fn build(self) -> Result<Tokenizer, String> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|error| format!("failed to create hf hub api: {error}"))?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            self.repo,
            hf_hub::RepoType::Model,
            self.revision,
        ));
        let tokenizer_file = repo.get(&self.file).map_err(|error| {
            format!("failed to download {} from HuggingFace: {error}", self.file)
        })?;
        let inner = tokenizers::Tokenizer::from_file(tokenizer_file)
            .map_err(|error| format!("failed to load tokenizer file: {error}"))?;
        Ok(Tokenizer { inner })
    }
}
