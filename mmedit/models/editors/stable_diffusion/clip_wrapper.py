import numpy as np
import torch
import torch.nn as nn
import os

from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
from mmengine.logging import MMLogger
from torch.hub import _get_torch_home
from mmengine.utils.dl_utils import load_url
from mmengine.utils.path import mkdir_or_exist

from transformers.models.clip.feature_extraction_clip import CLIPFeatureExtractor
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel


from mmengine.runner.checkpoint import CheckpointLoader
logger = MMLogger.get_current_instance()

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                images[idx] = np.zeros(images[idx].shape)  # black image

        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts


class TransformersCheckpointLoader:

    def load_from_cache_subdir(model_dir, model_name):

        if model_dir is None:
            torch_home = _get_torch_home()
            model_dir = os.path.join(torch_home, 'checkpoints', model_name)

        mkdir_or_exist(model_dir)


def load_clip_submodels(cached_folder, submodels, requires_safety_checker, loading_kwargs):
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(cached_folder, 'tokenizer'), **loading_kwargs)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(os.path.join(cached_folder, 'feature_extractor'), **loading_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(cached_folder, 'text_encoder'), **loading_kwargs)
    safety_checker = None
    if requires_safety_checker:
        submodels.append('safety_checker')
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(os.path.join(cached_folder, 'safety_checker'), **loading_kwargs)

    return tokenizer, feature_extractor, text_encoder, safety_checker
    