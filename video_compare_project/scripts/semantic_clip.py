# semantic_clip.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

# Lazy-load
_clip_model = None
_clip_proc = None

def get_clip_model(model_name="openai/clip-vit-base-patch32"):
    global _clip_model, _clip_proc
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_proc = CLIPProcessor.from_pretrained(model_name)
        # ensure eval mode
        _clip_model.eval()
    return _clip_model, _clip_proc

def describe_region_with_clip(pil_image, candidate_texts, top_k=1, device="cpu"):
    """
    Given a PIL image and a list of candidate text labels, returns top_k text labels with similarity scores.
    Uses CLIP zero-shot scoring (image-text similarity).
    """
    model, proc = get_clip_model()
    # prepare inputs
    inputs = proc(text=candidate_texts, images=pil_image, return_tensors="pt", padding=True)
    # move to cpu (model is CPU by default in our install)
    with torch.no_grad():
        outputs = model(**inputs)
        image_emb = outputs.image_embeds  # (1, d)
        text_emb = outputs.text_embeds    # (len(labels), d)
        # normalize
        image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        sims = (image_emb @ text_emb.T).squeeze(0).cpu().numpy()  # similarities
    # return top_k list of (label, score)
    idxs = np.argsort(-sims)[:top_k]
    return [(candidate_texts[i], float(sims[i])) for i in idxs]
