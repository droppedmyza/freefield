from __future__ import annotations

KLEIN = "flux-2-klein-4b-fp8.safetensors"
KLEIN_TE = "qwen_3_4b_fp8_mixed.safetensors"
KLEIN_VAE = "flux2-vae.safetensors"
WAN = "Wan2_2-TI2V-5B-Turbo-Q6_K.gguf"
WAN_TE = "umt5-xxl-encoder-Q5_K_M.gguf"
WAN_VAE = "wan2.2_vae.safetensors"
WAN14_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN14_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
WAN14_TE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN14_VAE = "wan_2.1_vae.safetensors"
LIGHTX_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
LIGHTX_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"


def node(class_type: str, title: str, **inputs) -> dict:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def keyframe_stage(g: dict, refs: list[str], width: int, height: int, prefix: str = "keyframe",
                   label: str = "", exact: tuple[str, ...] = ()) -> None:
    p = prefix
    t = (lambda title: f"{label}: {title}") if label else (lambda title: title)
    rid = (lambda ref, part: f"{ref}_{part}") if p == "keyframe" else (lambda ref, part: f"{p}_{ref}_{part}")
    g["klein_model"] = node("UNETLoader", "FLUX.2 klein 4B (Apache-2.0)", unet_name=KLEIN, weight_dtype="default")
    g["klein_te"] = node("CLIPLoader", "Qwen3-4B text encoder", clip_name=KLEIN_TE, type="flux2", device="default")
    g["klein_vae"] = node("VAELoader", "FLUX.2 VAE", vae_name=KLEIN_VAE)
    g[f"{p}_prompt"] = node("CLIPTextEncode", t("Keyframe prompt (what the still should show)"), text="", clip=["klein_te", 0])
    pos = [f"{p}_prompt", 0]
    neg_base = f"{p}_negative"
    g[neg_base] = node("ConditioningZeroOut", t("Empty negative (CFG 1)"), conditioning=[f"{p}_prompt", 0])
    neg = [neg_base, 0]
    for i, ref in enumerate(refs):
        if ref in exact:
            pixels = [ref, 0]
        else:
            g[rid(ref, "scaled")] = node("ImageScaleToTotalPixels", t(f"Reference {i + 1}: scale to ~1MP"),
                                      image=[ref, 0], upscale_method="lanczos", megapixels=1.0, resolution_steps=16)
            pixels = [rid(ref, "scaled"), 0]
        g[rid(ref, "latent")] = node("VAEEncode", t(f"Reference {i + 1}: encode" + (" at its own size" if ref in exact else "")),
                                     pixels=pixels, vae=["klein_vae", 0])
        g[rid(ref, "ref_pos")] = node("ReferenceLatent", t(f"Attach reference {i + 1} (positive)"), conditioning=pos, latent=[rid(ref, "latent"), 0])
        g[rid(ref, "ref_neg")] = node("ReferenceLatent", t(f"Attach reference {i + 1} (negative)"), conditioning=neg, latent=[rid(ref, "latent"), 0])
        pos, neg = [rid(ref, "ref_pos"), 0], [rid(ref, "ref_neg"), 0]
    g[f"{p}_guider"] = node("CFGGuider", t("Guider (CFG 1 for the distilled model)"), model=["klein_model", 0], positive=pos, negative=neg, cfg=1.0)
    g[f"{p}_noise"] = node("RandomNoise", t("Keyframe seed"), noise_seed=42)
    g[f"{p}_sampler_select"] = node("KSamplerSelect", t("Sampler"), sampler_name="euler")
    g[f"{p}_sigmas"] = node("Flux2Scheduler", t("4-step schedule"), steps=4, width=width, height=height)
    g[f"{p}_empty"] = node("EmptyFlux2LatentImage", t("Keyframe canvas"), width=width, height=height, batch_size=1)
    g[f"{p}_sample"] = node("SamplerCustomAdvanced", t("Generate keyframe"), noise=[f"{p}_noise", 0], guider=[f"{p}_guider", 0],
                                sampler=[f"{p}_sampler_select", 0], sigmas=[f"{p}_sigmas", 0], latent_image=[f"{p}_empty", 0])
    g[p] = node("VAEDecode", t("Keyframe image"), samples=[f"{p}_sample", 0], vae=["klein_vae", 0])
    g[f"{p}_save"] = node("SaveImage", t("Save keyframe (also used by the eval)"), images=[p, 0], filename_prefix=f"freefield/{p}")


def video_stage(g: dict, width: int, height: int, length: int, start: list | None = None) -> None:
    g["wan_model"] = node("UnetLoaderGGUF", "Wan 2.2 TI2V-5B Turbo Q6 (Apache-2.0)", unet_name=WAN)
    g["wan_te"] = node("CLIPLoaderGGUF", "umt5-xxl text encoder (GGUF)", clip_name=WAN_TE, type="wan")
    g["wan_vae"] = node("VAELoader", "Wan 2.2 VAE", vae_name=WAN_VAE)
    g["motion_prompt"] = node("CLIPTextEncode", "Motion prompt (what happens in the clip)", text="", clip=["wan_te", 0])
    g["motion_negative"] = node("ConditioningZeroOut", "Empty negative (CFG 1)", conditioning=["motion_prompt", 0])
    g["video_latent"] = node("Wan22ImageToVideoLatent", "Start the clip from the keyframe", vae=["wan_vae", 0],
                             width=width, height=height, length=length, batch_size=1, start_image=start or ["keyframe", 0])
    g["wan_shift"] = node("ModelSamplingSD3", "Sampling shift", model=["wan_model", 0], shift=8.0)
    g["video_sample"] = node("KSampler", "Animate (4 steps)", model=["wan_shift", 0], seed=42, steps=4, cfg=1.0,
                             sampler_name="euler", scheduler="simple", positive=["motion_prompt", 0],
                             negative=["motion_negative", 0], latent_image=["video_latent", 0], denoise=1.0)
    g["video_frames"] = node("VAEDecode", "Decode frames", samples=["video_sample", 0], vae=["wan_vae", 0])


def save_stage(g: dict, frames_ref: list, fps: float = 24.0) -> None:
    g["watermark"] = node("TextOverlay", "Watermark (made with FreeField)", images=frames_ref, text="FreeField · open source",
                          font_size=2.5, color="#ffffff", position="top", align="right", outline=True)
    g["video"] = node("CreateVideo", "Frames to video", images=["watermark", 0], fps=fps)
    g["clip_save"] = node("SaveVideo", "Save clip (the result)", video=["video", 0], filename_prefix="freefield/clip",
                          **{"format": "mp4", "format.codec": "h264"})


def wan14_models(g: dict) -> None:
    g["wan14_te"] = node("CLIPLoader", "umt5-xxl text encoder (fp8)", clip_name=WAN14_TE, type="wan", device="default")
    g["wan14_vae"] = node("VAELoader", "Wan 2.1 VAE (used by the 14B models)", vae_name=WAN14_VAE)
    for expert, unet, lora in (("high", WAN14_HIGH, LIGHTX_HIGH), ("low", WAN14_LOW, LIGHTX_LOW)):
        g[f"wan14_{expert}"] = node("UNETLoader", f"Wan 2.2 I2V-A14B {expert}-noise expert (fp8)", unet_name=unet, weight_dtype="default")
        g[f"wan14_{expert}_lora"] = node("LoraLoaderModelOnly", f"lightx2v 4-step LoRA ({expert} noise)",
                                         model=[f"wan14_{expert}", 0], lora_name=lora, strength_model=1.0)
        g[f"wan14_{expert}_shift"] = node("ModelSamplingSD3", f"Sampling shift ({expert})", model=[f"wan14_{expert}_lora", 0], shift=5.0)


def wan14_segment(g: dict, p: str, title: str, start: list, end: list | None, width: int, height: int, length: int) -> None:
    g[f"{p}_prompt"] = node("CLIPTextEncode", f"{title}: motion prompt", text="", clip=["wan14_te", 0])
    g[f"{p}_negative"] = node("ConditioningZeroOut", "Empty negative (CFG 1)", conditioning=[f"{p}_prompt", 0])
    dims = dict(vae=["wan14_vae", 0], width=width, height=height, length=length, batch_size=1, start_image=start)
    if end is None:
        g[f"{p}_latent"] = node("WanImageToVideo", f"{title}: pin the first frame", positive=[f"{p}_prompt", 0],
                                negative=[f"{p}_negative", 0], **dims)
    else:
        g[f"{p}_latent"] = node("WanFirstLastFrameToVideo", f"{title}: pin the first and last frame", positive=[f"{p}_prompt", 0],
                                negative=[f"{p}_negative", 0], end_image=end, **dims)
    common = dict(positive=[f"{p}_latent", 0], negative=[f"{p}_latent", 1], cfg=1.0, sampler_name="euler", scheduler="simple",
                  steps=4, noise_seed=42)
    g[f"{p}_sample_high"] = node("KSamplerAdvanced", f"{title}: motion layout, steps 0-2 (high-noise expert)",
                                 model=["wan14_high_shift", 0], latent_image=[f"{p}_latent", 2], add_noise="enable",
                                 start_at_step=0, end_at_step=2, return_with_leftover_noise="enable", **common)
    g[f"{p}_sample"] = node("KSamplerAdvanced", f"{title}: detail, steps 2-4 (low-noise expert)", model=["wan14_low_shift", 0],
                            latent_image=[f"{p}_sample_high", 0], add_noise="disable", start_at_step=2, end_at_step=10000,
                            return_with_leftover_noise="disable", **common)
    g[f"{p}_frames"] = node("VAEDecode", f"{title}: decode frames", samples=[f"{p}_sample", 0], vae=["wan14_vae", 0])
