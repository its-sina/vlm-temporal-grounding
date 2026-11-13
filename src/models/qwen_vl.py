"""Qwen-VL inference (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) with explicit, separately timed stages.

The runner splits generation into vision encoding, LLM prefill and greedy decoding so that

* precomputed or per-frame visual embeddings can replace the standard vision path (``vision_fn``),
* visual tokens can be dropped before the LLM (``keep_fn``, e.g. frame deduplication),
* M-RoPE positions can be rewritten (``pos_fn``) and token ids replaced for the embedding lookup
  only (``ids_fn``, e.g. delimiter replacement), and
* each stage can be timed with CUDA synchronisation.

Inputs are built like the reference pipeline: ``qwen_vl_utils.process_vision_info`` followed by
``AutoProcessor(use_fast=True)``, bf16 weights and FlashAttention-2. Decoding is greedy, matching
the checkpoints' generation configs (Qwen2-VL top_k=1; Qwen2.5-VL near-zero temperature with its
repetition penalty, which is applied here over prompt and generated ids as in Hugging Face).
``generate_reference`` runs ``model.generate`` for equivalence checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.representations import token_layout
from src.utils.timing import synchronized_time


@dataclass
class RunOutput:
    text: str
    n_input_tokens: int
    n_visual_tokens: int          # visual tokens fed to the LLM
    n_visual_tokens_encoded: int  # visual tokens produced by the vision encoder
    t_preprocess: float = 0.0
    t_vision: float = 0.0
    t_prefill: float = 0.0
    t_decode: float = 0.0
    n_generated: int = 0
    peak_mem_gb: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def t_model(self) -> float:
        return self.t_vision + self.t_prefill + self.t_decode

    @property
    def t_total(self) -> float:
        return self.t_preprocess + self.t_model


class QwenVLRunner:
    """Loads a Qwen2-VL / Qwen2.5-VL / Qwen3-VL checkpoint (local directory or Hub id)."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        from transformers import AutoConfig, AutoProcessor

        self.device = device
        self.model_type = AutoConfig.from_pretrained(model_path).model_type
        if self.model_type == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as model_cls
        elif self.model_type == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration as model_cls
        elif self.model_type == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration as model_cls
        else:
            raise ValueError(f"unsupported model type {self.model_type!r}")
        self.model = model_cls.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="flash_attention_2").eval()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.video_token_id = cfg.video_token_id
        gen_cfg = self.model.generation_config
        eos = gen_cfg.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.merge = self.model.model.visual.spatial_merge_size
        self.patch = int(self.model.model.visual.patch_size if self.model_type == "qwen3_vl"
                         else self.model.config.vision_config.patch_size)
        token_layout.set_patch_size(self.patch)
        # Qwen3-VL DeepStack: features of intermediate ViT layers are added to the hidden states of
        # the first LLM layers at the visual positions. encode_images returns them concatenated to the
        # main embedding along the feature dimension ([n, D * (1 + n_deepstack)]), so every per-token
        # row operation (scatter into cells, deduplication, reordering) carries them along.
        self.n_deepstack = (len(self.model.config.vision_config.deepstack_visual_indexes)
                            if self.model_type == "qwen3_vl" else 0)
        self.hidden = self.model.config.text_config.hidden_size if self.model_type == "qwen3_vl" else None
        self.rep_penalty = float(getattr(gen_cfg, "repetition_penalty", 1.0) or 1.0)

    # ------------------------------------------------------------------ inputs
    def prepare(self, messages: list[dict]):
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kw = {"image_patch_size": self.patch} if self.model_type == "qwen3_vl" else {}
        has_video = any(c.get("type") == "video" for m in messages for c in m["content"] if isinstance(c, dict))
        metas = None
        if self.model_type == "qwen3_vl" and has_video:
            # Qwen3-VL video needs the frame metadata (fps, indices) to write the textual timestamps.
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [messages], return_video_kwargs=True, return_video_metadata=True, **kw)
            if video_inputs is not None:
                video_inputs, metas = zip(*video_inputs)
                video_inputs, metas = list(video_inputs), list(metas)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, **kw)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                return_tensors="pt", video_metadata=metas, **video_kwargs)
        return inputs.to(self.device)

    # ------------------------------------------------------------------ vision
    @torch.no_grad()
    def encode_images(self, pixel_values, image_grid_thw) -> list[torch.Tensor]:
        """Per-image embeddings [n_tokens_i, D] in raster order of merged tokens
        (Qwen3-VL: [n_tokens_i, D * (1 + n_deepstack)], main | deepstack_0 | deepstack_1 | ...)."""
        out = self.model.model.get_image_features(pixel_values, image_grid_thw)
        if self.model_type != "qwen3_vl":
            return list(out)
        embeds, deepstack = out
        sizes = [e.shape[0] for e in embeds]
        return list(torch.cat([torch.cat(embeds), *deepstack], dim=-1).split(sizes))

    @torch.no_grad()
    def encode_pil(self, images) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Encode PIL images through the image processor (no text). Returns (embeddings, grid_thw)."""
        ip = self.processor.image_processor(images=images, return_tensors="pt")
        pv = ip["pixel_values"].to(self.device)
        thw = ip["image_grid_thw"].to(self.device)
        return self.encode_images(pv, thw), thw

    def split_deepstack(self, embeds: torch.Tensor, dtype) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Split concatenated Qwen3-VL embeddings into (main, deepstack list); no-op for other models."""
        if not self.n_deepstack:
            return embeds, None
        d = self.hidden
        deepstack = [embeds[:, d * (k + 1):d * (k + 2)].to(dtype) for k in range(self.n_deepstack)]
        return embeds[:, :d], deepstack

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def run(self, inputs, image_embeds: list[torch.Tensor] | None = None, keep_fn=None,
            max_new_tokens: int = 128, t_preprocess: float = 0.0, vision_fn=None, pos_fn=None,
            ids_fn=None) -> RunOutput:
        """Encode, prefill and greedily decode one prompt.

        image_embeds: precomputed per-image embeddings (skips the vision encoder).
        vision_fn:    callable(runner, inputs) -> per-image embeddings (timed as vision).
        keep_fn:      callable(input_ids, position_ids, inputs_embeds) -> bool mask [L] of the tokens
                      fed to the LLM (timed as prefill).
        pos_fn:       callable(input_ids, position_ids) -> new position ids, applied before keep_fn.
        ids_fn:       callable(input_ids) -> ids used for the embedding lookup only; positions, the
                      visual-token mask and the token count are computed from the original ids.
        """
        m = self.model.model
        torch.cuda.reset_peak_memory_stats(self.device)
        input_ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        has_img = "pixel_values" in inputs
        has_vid = "pixel_values_videos" in inputs

        t0 = synchronized_time()
        vid_embeds = None
        n_enc = 0
        if has_img:
            if image_embeds is None:
                image_embeds = vision_fn(self, inputs) if vision_fn else self.encode_images(
                    inputs["pixel_values"], inputs["image_grid_thw"])
            n_enc = sum(e.shape[0] for e in image_embeds) if image_embeds is not None else 0
        vid_deepstack = None
        if has_vid:
            vo = m.get_video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"])
            if self.model_type == "qwen3_vl":
                vo, vid_deepstack = vo
            vid_embeds = list(vo)
            n_enc += sum(e.shape[0] for e in vid_embeds)
        t1 = synchronized_time()

        emb = m.get_input_embeddings()(ids_fn(input_ids) if ids_fn is not None else input_ids)
        deepstack = None
        mask = None
        if has_img:
            ie = torch.cat(image_embeds).to(emb.dtype)
            mask = input_ids == self.image_token_id
            ie, deepstack = self.split_deepstack(ie, emb.dtype)
            emb[mask] = ie
        if has_vid:
            assert not (has_img and self.n_deepstack), "mixed image + video input is not supported for Qwen3-VL"
            ve = torch.cat(vid_embeds).to(emb.dtype)
            emb[input_ids == self.video_token_id] = ve
            if self.n_deepstack:   # video DeepStack, as in Qwen3VLModel.forward (video-only branch)
                deepstack = [d.to(emb.dtype) for d in vid_deepstack]
                mask = input_ids == self.video_token_id
        rope_kw = {"second_per_grid_ts": inputs.get("second_per_grid_ts")} if self.model_type == "qwen2_5_vl" else {}
        pos, _ = m.get_rope_index(input_ids, image_grid_thw=inputs.get("image_grid_thw"),
                                  video_grid_thw=inputs.get("video_grid_thw"), attention_mask=attn, **rope_kw)
        if pos_fn is not None:
            pos = pos_fn(input_ids[0], pos[:, 0]).unsqueeze(1).to(pos.dtype)
        if keep_fn is not None:
            keep = keep_fn(input_ids[0], pos[:, 0], emb[0])
            emb = emb[:, keep]
            pos = pos[:, :, keep]
            ids_kept = input_ids[:, keep]
            if deepstack is not None:
                kept_visual = keep[mask[0]]
                deepstack = [d[kept_visual] for d in deepstack]
        else:
            ids_kept = input_ids
        n_vis_llm = int(((ids_kept == self.image_token_id) | (ids_kept == self.video_token_id)).sum())
        next_pos = int(pos.max()) + 1

        from transformers import DynamicCache

        cache = DynamicCache()
        vis_tok = self.video_token_id if (has_vid and not has_img) else self.image_token_id
        ds_kw = ({"visual_pos_masks": ids_kept == vis_tok, "deepstack_visual_embeds": deepstack}
                 if deepstack is not None else {})
        out = m.language_model(inputs_embeds=emb, position_ids=pos, past_key_values=cache, use_cache=True, **ds_kw)
        seen = torch.zeros(self.model.lm_head.out_features, dtype=torch.bool, device=self.device)
        seen[input_ids[0]] = True

        def pick(logits):
            if self.rep_penalty != 1.0:
                lg = logits[0].float()
                lg = torch.where(seen & (lg > 0), lg / self.rep_penalty,
                                 torch.where(seen, lg * self.rep_penalty, lg))
                t = int(lg.argmax())
            else:
                t = int(logits.argmax(-1))
            seen[t] = True
            return t

        tok = pick(self.model.lm_head(out.last_hidden_state[:, -1]))
        t2 = synchronized_time()

        gen = [tok]
        length = emb.shape[1]
        while len(gen) < max_new_tokens and tok not in self.eos:
            p = torch.full((3, 1, 1), next_pos, device=self.device, dtype=pos.dtype)
            next_pos += 1
            e = m.get_input_embeddings()(torch.tensor([[tok]], device=self.device))
            cache_pos = torch.tensor([length], device=self.device)
            length += 1
            out = m.language_model(inputs_embeds=e, position_ids=p, past_key_values=cache,
                                   use_cache=True, cache_position=cache_pos)
            tok = pick(self.model.lm_head(out.last_hidden_state[:, -1]))
            gen.append(tok)
        t3 = synchronized_time()
        text = self.processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return RunOutput(text=text,
                         n_input_tokens=int(input_ids.shape[1]) if keep_fn is None else int(emb.shape[1]),
                         n_visual_tokens=n_vis_llm, n_visual_tokens_encoded=n_enc,
                         t_preprocess=t_preprocess, t_vision=t1 - t0, t_prefill=t2 - t1, t_decode=t3 - t2,
                         n_generated=len(gen),
                         peak_mem_gb=torch.cuda.max_memory_allocated(self.device) / 2**30)

    @torch.no_grad()
    def generate_reference(self, inputs, max_new_tokens: int = 128, greedy: bool = False) -> str:
        """Reference path: ``model.generate`` with the checkpoint generation config
        (``greedy=True`` forces do_sample=False, e.g. for Qwen3-VL whose config samples)."""
        kw = {"do_sample": False, "top_k": None, "top_p": None, "temperature": None} if greedy else {}
        ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kw)
        return self.processor.batch_decode([ids[0, inputs["input_ids"].shape[1]:]], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
