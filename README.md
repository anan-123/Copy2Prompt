# Copy2Prompt

Copy2Prompt is a novel framework that generates high-fidelity image variations while completely avoiding copyright infringement. By operating entirely in the textual domain, our three-stage pipeline converts reference images to detailed descriptions, enhances them through language models, and synthesizes new images without any pixel-level reuse.

## Key Features

- **Copyright Safe**: Zero pixel reuse - operates entirely through text descriptions
- **High Fidelity**: Maintains semantic similarity to source images (CLIP Score: 0.77-0.83)
- **Multi-Backend**: Support for DALL-E 3, Stable Diffusion XL, and custom models
- **RL Optimization**: Automated prompt enhancement using reinforcement learning
- **Flexible Pipeline**: Configurable prompt lengths and enhancement strategies


## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenAI API key (for GPT-4 and DALL-E 3)
- Hugging Face account (for model access)

## Pipeline Overview

Copy2Prompt consists of three main stages:

```
Source Image → [Vision-Language Model] → Initial Prompt
                        ↓
Initial Prompt → [Prompt Enhancement] → Enhanced Prompt  
                        ↓
Enhanced Prompt → [Text-to-Image Model] → Generated Image
```

### Stage 1: Image Understanding
- **Model**: LLaVA (Large Language and Vision Assistant)
- **Input**: Source image
- **Output**: Detailed textual description (800-2400 tokens)
- **Features**: Semantic content, spatial relationships, visual attributes

### Stage 2: Prompt Enhancement
- **Models**: GPT-4.0/4.1 or RL-based optimization
- **Input**: Initial textual description
- **Output**: Optimized prompt (77-1000 words)
- **Features**: Linguistic coherence, contextual expansion, diffusion model optimization

### Stage 3: Image Generation
- **Models**: DALL-E 3, Stable Diffusion XL with Compel
- **Input**: Enhanced textual prompt
- **Output**: Generated image variation
- **Features**: High-fidelity synthesis, copyright-free generation

## Usage Examples

### Basic Usage

```python
from copy2prompt import Copy2PromptPipeline

# Simple image variation
pipeline = Copy2PromptPipeline()
result = pipeline.generate("input.jpg", output_path="output.jpg")
```

### Advanced Configuration

```python
# Custom pipeline with specific models
pipeline = Copy2PromptPipeline(
    vision_model="llava-hf/llava-v1.6-mistral-7b-hf",
    enhancer_model="gpt-4.1", 
    generator="dall-e-3",
    target_length=200
)

# Generate with custom settings
result = pipeline.generate(
    image_path="input.jpg",
    target_length=500,
    enhancement_strategy="rl",
    output_path="output.jpg",
    return_metrics=True
)

print(f"CLIP Score: {result.metrics.clip_score}")
print(f"SSIM: {result.metrics.ssim}")
print(f"LPIPS: {result.metrics.lpips}")
```

### Batch Processing

```python
from copy2prompt import BatchProcessor

processor = BatchProcessor(
    input_dir="./input_images",
    output_dir="./generated_images",
    config_file="config.yaml"
)

results = processor.process_batch()
for result in results:
    print(f"{result.input_path} -> {result.output_path} (CLIP: {result.clip_score})")
```

### RL-Based Prompt Optimization

```python
from copy2prompt.rl import RLOptimizer

# Initialize RL agent
rl_optimizer = RLOptimizer(
    base_model="sdxl",
    reward_model="clip",
    max_iterations=10
)

# Optimize prompts
optimized_result = rl_optimizer.optimize(
    image_path="input.jpg",
    base_prompt="Sunset at the beach with a group of people",
    modifier_categories=["quality", "technical", "composition"]
)

print(f"Optimized prompt: {optimized_result.final_prompt}")
print(f"CLIP improvement: {optimized_result.improvement}")
```




## Evaluation

### Similarity Metrics

Copy2Prompt uses three primary metrics for evaluation:

- **CLIP Score**: Semantic similarity between text and image (0.77-0.83)
- **SSIM**: Structural similarity preserving layout and texture (0.16-0.22)
- **LPIPS**: Perceptual distance using deep features (0.66-0.70)


### Benchmark Results

| Method | CLIP Score | SSIM | LPIPS | Copyright Safe |
|--------|------------|------|-------|----------------|
| SDXL I2I Baseline | 0.92 | 0.62 | 0.16 | ❌ |
| Copy2Prompt (DALL-E 3) | 0.81 | 0.20 | 0.70 | ✅ |
| Copy2Prompt (SDXL+C) | 0.79 | 0.20 | 0.68 | ✅ |
| BLIP-2 → SDXL | 0.66 | 0.11 | 0.75 | ✅ |

## Results

### Performance Highlights

- **9.5% CLIP Score Improvement**: RL optimization achieves 0.7515 → 0.8228
- **Multi-Backend Support**: Consistent performance across DALL-E 3 and SDXL
- **Prompt Length Optimization**: 77-word prompts perform competitively with longer variants
- **Copyright Compliance**: Zero pixel reuse while maintaining semantic fidelity

### Qualitative Results

The framework successfully generates semantically similar images across various domains:
- Landscape photography with preserved composition and lighting
- Portrait generation maintaining facial structure and styling
- Architectural images preserving geometric relationships
- Abstract art maintaining color schemes and artistic elements

## API Reference

### Copy2PromptPipeline

```python
class Copy2PromptPipeline:
    def __init__(
        self,
        vision_model: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        enhancer_model: str = "gpt-4",
        generator: str = "sdxl",
        device: str = "auto"
    )
    
    def generate(
        self,
        image_path: str,
        target_length: int = 77,
        enhancement_strategy: str = "llm",
        output_path: str = None,
        return_metrics: bool = False
    ) -> PipelineResult
```

### RLOptimizer

```python
class RLOptimizer:
    def __init__(
        self,
        base_model: str = "sdxl",
        reward_model: str = "clip",
        max_iterations: int = 10
    )
    
    def optimize(
        self,
        image_path: str,
        base_prompt: str,
        modifier_categories: List[str]
    ) -> OptimizationResult
```
=
