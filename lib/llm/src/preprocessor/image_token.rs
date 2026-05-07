// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve a model's image-placeholder token id (e.g. `<|image_pad|>`,
//! `<image>`, `[IMG]`, `<|media_pad|>`) for MM-aware KV routing.
//!
//! Three tiers, each anchored to a verified HF convention:
//!
//!   1. Numeric ID in `config.json` — Qwen-VL family (`image_token_id`),
//!      PaliGemma (`image_token_index`), Kimi-K2.5 (`media_placeholder_token_id`).
//!   2. String field in `processor_config.json` or `tokenizer_config.json`
//!      (`image_token`) — LLaVA, LLaVA-Next, Pixtral, Llama-4-Vision.
//!   3. Vocab probe of common literal strings — Phi-3-V (token in vocab but
//!      not exposed in any `image_token` field), and a backstop for new
//!      models that follow one of the well-known placeholder conventions.
//!
//! Returns `None` for models that don't expose the placeholder via any of
//! these channels (e.g. Phi-4-multimodal-instruct, which resolves it at
//! runtime via a custom `Phi4MMProcessor` requiring `trust_remote_code`).
//! Callers should treat `None` as "MM-aware routing disabled for this model"
//! and fall back to text-prefix routing — same behavior as today.

use std::path::Path;

use serde::Deserialize;

use crate::tokenizers::TokenIdType;
use crate::tokenizers::traits::Tokenizer;

/// Encode `s` and return its token id iff it tokenizes to exactly one token
/// (i.e., it's a registered single special-token in the vocab). Returns
/// `None` otherwise — that's the signal the placeholder isn't actually
/// reachable from this tokenizer.
fn token_to_id(tokenizer: &dyn Tokenizer, s: &str) -> Option<TokenIdType> {
    let enc = tokenizer.encode(s).ok()?;
    let ids = enc.token_ids();
    if ids.len() == 1 { Some(ids[0]) } else { None }
}

/// Tier-3 vocab-probe list. Order matters: more specific / less ambiguous
/// strings first. First match wins.
const COMMON_PLACEHOLDERS: &[&str] = &[
    "<|image_pad|>", // Qwen2-VL / Qwen3-VL (also covered by tier 1)
    "<|media_pad|>", // Kimi-K2.5 (also covered by tier 1)
    "<image>",       // LLaVA / LLaVA-Next / Idefics2/3 / Mllama (also tier 2)
    "[IMG]",         // Pixtral (also tier 2)
    "<IMG_CONTEXT>", // InternVL
    "<|image|>",     // Phi-3-V (only tier that catches it)
];

#[derive(Deserialize, Default)]
struct ConfigJson {
    image_token_id: Option<i64>,
    image_token_index: Option<i64>,
    media_placeholder_token_id: Option<i64>,
}

#[derive(Deserialize, Default)]
struct ImageTokenStr {
    image_token: Option<String>,
}

/// Resolve the image-placeholder token id for a model on disk.
///
/// `model_dir` is the local model directory (typically the HF snapshot dir
/// containing `config.json`, `tokenizer.json`, etc.).
///
/// Returns `None` when no tier produces a hit. Callers should treat this as
/// "MM-aware routing disabled for this model".
pub fn resolve_image_token_id(
    model_dir: &Path,
    tokenizer: &dyn Tokenizer,
) -> Option<TokenIdType> {
    // Tier 1: numeric ID in config.json
    if let Some(cfg) = load_json::<ConfigJson>(model_dir, "config.json")
        && let Some(id) = cfg
            .image_token_id
            .or(cfg.image_token_index)
            .or(cfg.media_placeholder_token_id)
        && id >= 0
    {
        tracing::info!(
            target: "lightseek_mm",
            image_token_id = id,
            source = "config.json::image_token_id|image_token_index|media_placeholder_token_id",
            "resolved image-placeholder token id (tier 1, numeric)"
        );
        return Some(id as TokenIdType);
    }

    // Tier 2: string in processor_config.json or tokenizer_config.json
    for filename in ["processor_config.json", "tokenizer_config.json"] {
        let Some(cfg) = load_json::<ImageTokenStr>(model_dir, filename) else {
            continue;
        };
        let Some(s) = cfg.image_token else { continue };
        if let Some(id) = token_to_id(tokenizer, &s) {
            tracing::info!(
                target: "lightseek_mm",
                image_token = %s,
                image_token_id = id,
                source = filename,
                "resolved image-placeholder token id (tier 2, string field)"
            );
            return Some(id);
        }
        tracing::warn!(
            target: "lightseek_mm",
            image_token = %s,
            source = filename,
            "HF config reported image_token but tokenizer doesn't \
             map it to a single id; trying next tier"
        );
    }

    // Tier 3: vocab probe of common placeholders
    for placeholder in COMMON_PLACEHOLDERS {
        if let Some(id) = token_to_id(tokenizer, placeholder) {
            tracing::info!(
                target: "lightseek_mm",
                image_token = placeholder,
                image_token_id = id,
                "resolved image-placeholder token id (tier 3, vocab probe)"
            );
            return Some(id);
        }
    }

    // Caller in OpenAIPreprocessor::new_with_parts emits a coalesced
    // per-model warn that names the model and reasons; we keep only a
    // debug breadcrumb here for ops who want the model_dir context.
    tracing::debug!(
        target: "lightseek_mm",
        model_dir = %model_dir.display(),
        "no tier produced an image-placeholder token id"
    );
    None
}

fn load_json<T: for<'de> Deserialize<'de>>(model_dir: &Path, filename: &str) -> Option<T> {
    let path = model_dir.join(filename);
    if !path.exists() {
        return None;
    }
    let bytes = std::fs::read(&path).ok()?;
    match serde_json::from_slice::<T>(&bytes) {
        Ok(v) => Some(v),
        Err(err) => {
            tracing::debug!(
                target: "lightseek_mm",
                file = filename,
                error = %err,
                "failed to parse JSON; skipping tier"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Stub tokenizer used to drive Tier-2/3 unit tests deterministically.
    /// Maps a fixed set of input strings to single-token ids; everything else
    /// produces a multi-token (rejected) encoding.
    struct StubTokenizer(std::collections::HashMap<String, crate::tokenizers::TokenIdType>);

    impl crate::tokenizers::traits::Encoder for StubTokenizer {
        fn encode(&self, input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            if let Some(&id) = self.0.get(input) {
                Ok(crate::tokenizers::Encoding::Sp(vec![id]))
            } else {
                Ok(crate::tokenizers::Encoding::Sp(vec![0u32, 1u32]))
            }
        }
        fn encode_batch(
            &self,
            inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            inputs.iter().map(|s| self.encode(s)).collect()
        }
    }

    impl crate::tokenizers::traits::Decoder for StubTokenizer {
        fn decode(
            &self,
            _ids: &[crate::tokenizers::TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<crate::tokenizers::traits::DecodeResult> {
            Ok(crate::tokenizers::traits::DecodeResult::Complete(String::new()))
        }
    }

    impl crate::tokenizers::traits::Tokenizer for StubTokenizer {}

    fn write_json(dir: &std::path::Path, name: &str, body: &str) {
        fs::write(dir.join(name), body).unwrap();
    }

    /// Tier 1: numeric `image_token_id` in config.json (Qwen2/2.5/3-VL family).
    #[test]
    fn tier1_image_token_id() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(
            tmp.path(),
            "config.json",
            r#"{"image_token_id": 151655}"#,
        );
        let tk = StubTokenizer(Default::default());
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), Some(151655));
    }

    /// Tier 1: alternate numeric field `image_token_index` (LLaVA-style).
    #[test]
    fn tier1_image_token_index() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(
            tmp.path(),
            "config.json",
            r#"{"image_token_index": 32000}"#,
        );
        let tk = StubTokenizer(Default::default());
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), Some(32000));
    }

    /// Tier 1: Kimi-K2.5's distinct field name.
    #[test]
    fn tier1_media_placeholder_token_id() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(
            tmp.path(),
            "config.json",
            r#"{"media_placeholder_token_id": 163605}"#,
        );
        let tk = StubTokenizer(Default::default());
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), Some(163605));
    }

    /// Tier 1: negative ids (ignored) → falls through.
    #[test]
    fn tier1_negative_id_falls_through() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(tmp.path(), "config.json", r#"{"image_token_id": -1}"#);
        let tk = StubTokenizer(Default::default());
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), None);
    }

    /// Tier 2: string `image_token` field in processor_config.json maps via tokenizer.
    #[test]
    fn tier2_image_token_string_processor_config() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(tmp.path(), "config.json", r#"{}"#);
        write_json(
            tmp.path(),
            "processor_config.json",
            r#"{"image_token": "[IMG]"}"#,
        );
        let mut map = std::collections::HashMap::new();
        map.insert("[IMG]".to_string(), 12345u32);
        let tk = StubTokenizer(map);
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), Some(12345));
    }

    /// Tier 2 fallback to tokenizer_config.json when processor_config.json absent.
    #[test]
    fn tier2_image_token_string_tokenizer_config() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(tmp.path(), "config.json", r#"{}"#);
        write_json(
            tmp.path(),
            "tokenizer_config.json",
            r#"{"image_token": "<image>"}"#,
        );
        let mut map = std::collections::HashMap::new();
        map.insert("<image>".to_string(), 32000u32);
        let tk = StubTokenizer(map);
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), Some(32000));
    }

    /// Tier 3: vocab probe catches placeholders even without a JSON field
    /// (Phi-3-V case).
    #[test]
    fn tier3_vocab_probe() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(tmp.path(), "config.json", r#"{}"#);
        let mut map = std::collections::HashMap::new();
        map.insert("<|image|>".to_string(), 32044u32);
        let tk = StubTokenizer(map);
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), Some(32044));
    }

    /// Empty model dir: every tier misses → graceful None.
    #[test]
    fn empty_dir_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let tk = StubTokenizer(Default::default());
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), None);
    }

    /// Tier 1 takes precedence over Tier 2 (numeric wins when both present).
    #[test]
    fn tier1_precedes_tier2() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(tmp.path(), "config.json", r#"{"image_token_id": 999}"#);
        write_json(
            tmp.path(),
            "processor_config.json",
            r#"{"image_token": "[IMG]"}"#,
        );
        let mut map = std::collections::HashMap::new();
        map.insert("[IMG]".to_string(), 111u32);
        let tk = StubTokenizer(map);
        // Numeric tier 1 wins
        assert_eq!(resolve_image_token_id(tmp.path(), &tk), Some(999));
    }
}
