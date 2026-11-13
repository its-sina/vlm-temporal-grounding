"""InternVL3.5 (Hugging Face ``InternVLForConditionalGeneration``) runner for delimiter experiments.

Differences from Qwen-VL that matter here:

* the InternViT encoder works on fixed 448 x 448 tiles (pixel shuffle -> 256 tokens per tile); a
  large image is cut into up to 12 tiles plus a thumbnail (``crop_to_patches``);
* the LLM uses 1-D RoPE: every token, visual or not, takes the next integer position;
* each image is wrapped as ``<img> <IMG_CONTEXT>*256n </img> \\n``, so consecutive images are
  separated by the three delimiter tokens ``</img> \\n <img>``.

Variants of ``run`` (all built from the processor output of a per-frame sequence):

  native   unmodified input (e.g. grid images processed by InternVL's own tiling)
  sep      per-frame sequence as built (delimiters present)
  neutral  the inner ``</img> \\n <img>`` ids replaced by three ``\\n`` ids for the embedding lookup
           only (same positions, same token count)
  gap      inner delimiters removed, the remaining tokens keep their original positions
  block    inner delimiters removed, positions renumbered 0..L-1
  raster   block + each window's visual tokens reordered into grid raster order
"""

from __future__ import annotations

import torch

from src.models.qwen_vl import RunOutput
from src.utils.timing import synchronized_time

VARIANTS = ("native", "sep", "neutral", "gap", "block", "raster")


def _images(messages):
    return [c["image"] for m in messages for c in m["content"] if c["type"] == "image"]


class InternVLRunner:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        from transformers import AutoProcessor, InternVLForConditionalGeneration

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = InternVLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device).eval()
        tok = self.processor.tokenizer
        self.image_token_id = self.model.config.image_token_id
        self.img_start, self.img_end = tok.convert_tokens_to_ids([self.processor.start_image_token,
                                                                  self.processor.end_image_token])
        nl = tok.encode("\n", add_special_tokens=False)
        assert len(nl) == 1, nl
        self.nl = nl[0]
        self.eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        self.model_type = "internvl"

    # ------------------------------------------------------------------ inputs
    def prepare(self, messages, crop_to_patches: bool):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], images=_images(messages), crop_to_patches=crop_to_patches,
                             return_tensors="pt")
        return inp.to(self.device)

    def inner_separators(self, ids: torch.Tensor) -> torch.Tensor:
        """Mask of the ``</img> \\n <img>`` triples directly between two images (no text in between)."""
        c, m = self.image_token_id, torch.zeros_like(ids, dtype=torch.bool)
        hit = ((ids[:-4] == c) & (ids[1:-3] == self.img_end) & (ids[2:-2] == self.nl)
               & (ids[3:-1] == self.img_start) & (ids[4:] == c))
        i = torch.nonzero(hit).flatten() + 1
        for d in range(3):
            m[i + d] = True
        return m

    @torch.no_grad()
    def encode(self, pixel_values) -> torch.Tensor:
        out = []
        for i in range(0, pixel_values.shape[0], 64):
            out.append(self.model.model.get_image_features(pixel_values[i:i + 64]))
        return torch.cat(out)  # [n_tiles, 256, D]

    # ------------------------------------------------------------------ run
    @torch.no_grad()
    def run(self, inputs, variant: str = "sep", raster=None, max_new_tokens: int = 128,
            t_preprocess: float = 0.0) -> RunOutput:
        """raster: for variant ``raster``, a callable(n_visual) -> permutation of the visual tokens."""
        if variant not in VARIANTS:
            raise ValueError(variant)
        m = self.model.model
        torch.cuda.reset_peak_memory_stats(self.device)
        ids = inputs["input_ids"][0]
        t0 = synchronized_time()
        feats = self.encode(inputs["pixel_values"])
        n_enc = feats.shape[0] * feats.shape[1]
        t1 = synchronized_time()
        emb_ids = ids
        inner = self.inner_separators(ids) if variant != "native" else torch.zeros_like(ids, dtype=torch.bool)
        if variant == "neutral":
            emb_ids = ids.clone()
            emb_ids[inner] = self.nl
        emb = m.get_input_embeddings()(emb_ids[None])
        vis = ids == self.image_token_id
        assert int(vis.sum()) == n_enc, (int(vis.sum()), n_enc)
        f = feats.reshape(-1, feats.shape[-1]).to(emb.dtype)
        if variant == "raster":
            f = f[raster(f.shape[0]).to(f.device)]
        emb[0, vis] = f
        pos = torch.arange(len(ids), device=ids.device)
        if variant in ("gap", "block", "raster"):
            keep = ~inner
            emb, ids_k = emb[:, keep], ids[keep]
            pos = pos[keep] if variant == "gap" else torch.arange(int(keep.sum()), device=ids.device)
        else:
            ids_k = ids
        n_vis_llm = int((ids_k == self.image_token_id).sum())

        from transformers import DynamicCache

        cache = DynamicCache()
        lm = m.language_model
        out = lm(inputs_embeds=emb, position_ids=pos[None], past_key_values=cache, use_cache=True)
        tok = int(self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
        t2 = synchronized_time()
        gen, length, nxt = [tok], emb.shape[1], int(pos.max()) + 1
        while len(gen) < max_new_tokens and tok not in self.eos:
            e = m.get_input_embeddings()(torch.tensor([[tok]], device=self.device))
            out = lm(inputs_embeds=e, position_ids=torch.tensor([[nxt]], device=self.device), past_key_values=cache,
                     use_cache=True, cache_position=torch.tensor([length], device=self.device))
            length += 1
            nxt += 1
            tok = int(self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
            gen.append(tok)
        t3 = synchronized_time()
        text = self.processor.batch_decode([gen], skip_special_tokens=True)[0]
        return RunOutput(text=text, n_input_tokens=int(emb.shape[1]), n_visual_tokens=n_vis_llm,
                         n_visual_tokens_encoded=n_enc, t_preprocess=t_preprocess, t_vision=t1 - t0,
                         t_prefill=t2 - t1, t_decode=t3 - t2, n_generated=len(gen),
                         peak_mem_gb=torch.cuda.max_memory_allocated(self.device) / 2**30,
                         extra={"n_inner_sep_tokens": int(inner.sum())})

    @torch.no_grad()
    def generate_reference(self, inputs, max_new_tokens: int = 128) -> str:
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def raster_permutation(layout, owner: dict, tok_side: int = 16):
    """Permutation that reorders the visual tokens of a per-frame sequence (frames in chronological
    order, ``tok_side x tok_side`` tokens each, each frame once in window ``owner[f]``) into grid raster
    order within each window: key = (cell_row, token_row, cell_col, token_col)."""
    cols = layout.cfg.cols
    per = tok_side * tok_side
    keys, base = [], 0
    for w in layout.windows:
        mine = [f for f in w.frames if owner[f] == w.index]
        block = []
        for j, f in enumerate(mine):
            ci = w.cells.index(f)
            for t in range(per):
                ty, tx = divmod(t, tok_side)
                block.append(((ci // cols, ty, ci % cols, tx), base + j * per + t))
        block.sort()
        keys += [i for _, i in block]
        base += len(mine) * per
    perm = torch.tensor(keys)

    def permutation(n: int) -> torch.Tensor:
        if n != len(perm):
            raise AssertionError((n, len(perm)))
        return perm
    return permutation
