"""Watermark removal orchestration pipeline."""

# Standalone pipeline (recommended - no ComfyUI dependency)
from .standalone import (
    StandaloneWatermarkJob,
    PipelineConfig,
    PipelineResult,
    run_pipeline
)

__all__ = [
    "StandaloneWatermarkJob",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]

# Legacy ComfyUI pipeline (optional - import only if needed)
# from .job import SoraWatermarkJob, PipelineConfig as LegacyPipelineConfig, PipelinePaths
