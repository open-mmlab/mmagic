from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel
from transformers.models.clip import CLIPTokenizer
from torch.utils.data import DataLoader
from personalized_dataset import PersonalizedBase
from customed_models import set_vico_modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from einops import rearrange
import os


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def get_input(batch, k):
    x = batch[k]
    if len(x.shape) == 3:
        x = x[..., None]
    x = rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    return x


def save_embeddings(text_encoder, placeholder_token_ids, placeholder_token, save_path):
    print("Saving embeddings")
    learned_embeds = (
        text_encoder
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


if __name__ == "__main__":
    save_steps = 99
    lambda_reg = 5e-4
    placeholder_token = "S*"
    num_vectors = 1
    output_dir = "vico/experiments/wooden_pot/"
    
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

    # set vico model
    set_vico_modules(unet)

    dataset = PersonalizedBase(
        data_root="vico_data/wooden_pot",
        size=512,
        set="train",
        repeats=100,
        per_image_tokens=False,
        init_text="pot"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=1)

    # Add the placeholder token in tokenizer
    placeholder_tokens = [placeholder_token]
    initializer_token = dataset.init_text

    if num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, num_vectors):
        additional_tokens.append(f"{placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()


    unet.requires_grad_(False)
    vae.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    def return_and_unfreeze_imageca_params(unet):
        params_to_return = []
        for name, layer in unet.named_modules():
            if name == "image_cross_attention":
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
                    params_to_return.append(layer.parameters())
        return params_to_return
    imageca_params = return_and_unfreeze_imageca_params(unet)

    params_to_optimize = [{"params": imageca_params, 'lr': 1e-5}, {"params": text_encoder.get_input_embeddings().parameters(), 'lr': 0.01}]

    optimizer = AdamW(
        params_to_optimize,
        lr=1e-5
    )
    
    unet.to("cuda")
    vae.to("cuda")
    text_encoder.to("cuda")
    # keep original embeddings as reference
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()
    global_step = 0

    for batch in dataloader:
        image = get_input(batch, "image")
        image = image.to("cuda")
        model_input = vae.encode(image).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor

        image_reference = get_input(batch, "image_ref")
        image_reference = image_reference.to("cuda")
        image_reference_latent = vae.encode(image_reference).latent_dist.sample()
        image_reference_latent = image_reference_latent * vae.config.scaling_factor

        c = batch["caption"]
        ph_pos = batch["placeholder_pos"]
        t_init = batch["text_init"]
        
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        text_inputs = tokenize_prompt(tokenizer, t_init, tokenizer_max_length=77)
        encoder_hidden_states = encode_prompt(text_encoder, text_inputs.input_ids, None)
        noisy_model_input = torch.cat([noisy_model_input, image_reference_latent], dim=0)
        model_pred, loss_reg = unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states,
            placeholder_position=ph_pos
        )
        target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") + lambda_reg * loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
    
        # Let's make sure we don't update any embedding weights besides the newly added token
        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
        index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[
                index_no_updates
            ] = orig_embeds_params[index_no_updates]

        global_step += 1
        if global_step % save_steps == 0:
            save_path = os.path.join(output_dir, f"learned_embeds-steps-{global_step}.bin")
            save_embeddings(text_encoder, placeholder_token_ids, placeholder_token, save_path)
            ca_state_dict = {k: v for k, v in unet.state_dict().items() if "image_cross_attention" in k}
            torch.save(ca_state_dict, os.path.join(output_dir, f"cross_attention-steps-{global_step}.bin"))
            