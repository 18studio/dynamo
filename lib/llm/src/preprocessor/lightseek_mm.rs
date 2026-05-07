// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust per-image token-count via the `llm-multimodal` crate.
//! Compiled only when the `lightseek-mm` cargo feature is enabled.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use llm_multimodal::{ImagePreProcessor, ImageProcessorRegistry, PreProcessorConfig};

// `find()` returns a `&dyn ImagePreProcessor` borrowed from the registry, so
// the registry must outlive every counter — `LazyLock` gives it `'static`.
static REGISTRY: LazyLock<ImageProcessorRegistry> =
    LazyLock::new(ImageProcessorRegistry::with_defaults);

/// Maps `(width, height) → num_image_tokens` for a single model using the
/// model's HF `preprocessor_config.json`.
pub struct LightseekMmCounter {
    processor: &'static dyn ImagePreProcessor,
    config: PreProcessorConfig,
    model_id: String,
}

impl LightseekMmCounter {
    /// Returns `Err` when `preprocessor_config.json` is missing or unparseable
    /// or no registered processor matches `model_id` / `model_type`. Callers
    /// should treat the error as "MM-aware routing disabled for this model"
    /// rather than failing the request.
    pub fn try_new(
        model_id: &str,
        model_type: Option<&str>,
        model_dir: &Path,
    ) -> Result<Self> {
        let cfg_path = model_dir.join("preprocessor_config.json");
        let json = std::fs::read_to_string(&cfg_path).with_context(|| {
            format!(
                "lightseek: failed to read preprocessor_config.json at {}",
                cfg_path.display()
            )
        })?;
        let config = PreProcessorConfig::from_json(&json).with_context(|| {
            format!(
                "lightseek: failed to parse preprocessor_config.json at {}",
                cfg_path.display()
            )
        })?;

        let processor = REGISTRY.find(model_id, model_type).ok_or_else(|| {
            anyhow!(
                "lightseek: no image processor registered for model_id={:?} model_type={:?}",
                model_id,
                model_type
            )
        })?;

        Ok(Self {
            processor,
            config,
            model_id: model_id.to_string(),
        })
    }

    pub fn count_tokens(&self, width: u32, height: u32) -> usize {
        self.processor
            .calculate_num_tokens(width, height, &self.config)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}
