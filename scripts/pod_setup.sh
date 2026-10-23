#!/usr/bin/env bash
set -euo pipefail
W=/workspace
C=$W/ComfyUI
COMFY_COMMIT=bdc63e72f5ab29fde448c161db3057827148c0ef
mkdir -p $W/logs
cd $W
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }
if [ ! -d $C ]; then
  git clone -q https://github.com/comfyanonymous/ComfyUI.git $C
  git -C $C checkout -q $COMFY_COMMIT
fi
[ -d $C/custom_nodes/ComfyUI-GGUF ] || git clone -q --depth 1 https://github.com/city96/ComfyUI-GGUF.git $C/custom_nodes/ComfyUI-GGUF
pip install -q -r $C/requirements.txt -r $C/custom_nodes/ComfyUI-GGUF/requirements.txt "comfy-api-proxy==0.1.6" 2>&1 | tail -3

M=$C/models
HF=https://huggingface.co
get() {
  [ -s "$M/$1" ] && return 0
  mkdir -p "$(dirname "$M/$1")"
  curl -sL --retry 5 --retry-delay 3 -o "$M/$1.part" "$2" && mv "$M/$1.part" "$M/$1"
}
get diffusion_models/flux-2-klein-4b-fp8.safetensors $HF/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors &
get text_encoders/qwen_3_4b_fp8_mixed.safetensors $HF/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors &
get vae/flux2-vae.safetensors $HF/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors &
R=$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files
[ "${PROFILE:-studio}" = lite ] || get diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors $R/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors &
[ "${PROFILE:-studio}" = lite ] || get diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors $R/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors &
[ "${PROFILE:-studio}" = lite ] || get vae/wan_2.1_vae.safetensors $R/vae/wan_2.1_vae.safetensors &
[ "${PROFILE:-studio}" = lite ] || get loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors $R/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors &
[ "${PROFILE:-studio}" = lite ] || get loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors $R/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors &
[ "${PROFILE:-studio}" = lite ] || get text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $HF/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors &
get diffusion_models/Wan2_2-TI2V-5B-Turbo-Q6_K.gguf $HF/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF/resolve/main/Wan2_2-TI2V-5B-Turbo-Q6_K.gguf &
get text_encoders/umt5-xxl-encoder-Q5_K_M.gguf $HF/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf &
get vae/wan2.2_vae.safetensors $R/vae/wan2.2_vae.safetensors &
get background_removal/birefnet.safetensors $HF/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors &
wait
du -sh $M/*/ | sort -h
ls -la $M/diffusion_models $M/loras 2>/dev/null || true

if ! curl -sf 127.0.0.1:8188/system_stats >/dev/null; then
  (cd $C && nohup python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-async-offload > $W/logs/comfy.log 2>&1 &)
  for _ in $(seq 1 120); do curl -sf 127.0.0.1:8188/system_stats >/dev/null && break; sleep 1; done
fi
if ! curl -sf 127.0.0.1:8189/api/v2/health >/dev/null; then
  nohup comfy-api-proxy --comfyui http://127.0.0.1:8188 --port 8189 --comfyui-base-dir $C --state-dir $W/proxy-state \
    --max-upload-mb 50 > $W/logs/proxy.log 2>&1 &
  for _ in $(seq 1 30); do curl -sf 127.0.0.1:8189/api/v2/health >/dev/null && break; sleep 1; done
fi
curl -s 127.0.0.1:8188/system_stats | cut -c1-600
curl -s 127.0.0.1:8189/api/v2/health; echo
