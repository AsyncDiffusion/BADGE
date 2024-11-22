# Biased to Balanced: Adaptive Guidance for Fair Text-to-Image Generation in Language and Visual Spaces

Official Implementation of the **Biased to Balanced: Adaptive Guidance for Fair Text-to-Image Generation in Language and Visual Spaces**. 

![example images](docs/images/intersection.jpg)

> Social biases in diffusion-based text-to-image models have sparked significant research into effective debiasing methods. Unfortunately, existing efforts either consider bias only from language (i.e., CLIP) or visual space (i.e., U-Net): (1) Debiasing only language space faces the issue of fair text condition fails to control visual space. Even when text conditions are uniformly distributed across attributes, the synthesized images can still display pseudo or over-correction, which points to potential biases within the visual space during the denoising process. (2) While only debiasing the visual space can arise where the biases inclination in the language and visual space are misaligned, leading to conflicts during the guidance process that can degrade the quality of synthesized images. As a response to above issues, we propose Biased to Balanced: Adaptive Guidance (BBA) that manipulates bias in language and visual (bimodal) spaces. To realize this, we first locate and quantify biases degree towards various attributes within bimodal space; then, during inference, BBA apply adaptive guidance in bimodal spaces based on the detected bias. BBA presents a self-debiasing framework that avoids extra training costs and corpora. Experiments demonstrate that BBA significantly improves the fairness in various intra- and inter-categories biases (e.g., gender, skin tone, age) while achieving considerable image fidelity.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### A Simple Example for Debiasing Text-to-Image Models.
The code aims to remove the gender bias of [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1).

Flags:
  - `--cls`: select the target class, e.g., doctor.
  - `--lam`: hyperparameter lambda of debiasing algorithm


For instance, to reproduce the experiments, run
```bash
cd generative
python main.py --cls doctor --lam 500
```

### Semantic Guidance (Using SD v1.5 as an example)
```python
from diffusers import SemanticStableDiffusionPipeline
device = 'cuda'
pipe = SemanticStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)
import torch
gen = torch.Generator(device=device)

gen.manual_seed(21)
out = pipe(prompt='a photo of the face of a woman', generator=gen, num_images_per_prompt=1, guidance_scale=7,
           editing_prompt=['smiling, smile',       # Concepts to apply 
                           'glasses, wearing glasses', 
                           'curls, wavy hair, curly hair', 
                           'beard, full beard, mustache'],
           reverse_editing_direction=[False, False, False, False], # Direction of guidance i.e. increase all concepts
           edit_warmup_steps=[10, 10, 10,10], # Warmup period for each concept
           edit_guidance_scale=[4, 5, 5, 5.4], # Guidance scale for each concept
           edit_threshold=[0.99, 0.975, 0.925, 0.96], # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
           edit_momentum_scale=0.3, # Momentum scale that will be added to the latent guidance
           edit_mom_beta=0.6, # Momentum beta
           edit_weights=[1,1,1,1,1] # Weights of the individual concepts against each other
          )
images = out.images
```

### Debiasing Zero-Shot Models.

Flags:
  - `--dataset`: select the dataset (waterbirds/celebA)
  - `--load_base_model`: select backbone model (clip_RN50/clip_ViTL14)
  - `--debias`: debias the text embedding or not
  - `--lam`: hyperparameter lambda of debiasing algorithm


For instance, to reproduce the experiments, run
```bash
cd discriminative
python main.py --dataset waterbirds --load_base_model clip_RN50 --debias
```

