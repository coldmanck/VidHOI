import logging
import os
import pickle
import numpy as np
from termcolor import colored
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.utils.logger import setup_logger

logger = setup_logger(name=__name__)


def load_semantic_embeddings(semantic_corpus, classes, precomputed_semantic_embs=None):
    """
    Load precomputed semantic embeddings if it exists. Otherwise, extract it from corpus.
    Args:
        semantic_corpus (str)
        classes (List[str])
        precomputed_semantic_embs (str)
    Returns:
        class_embs_dict (Dict[str: np.array])
    """
    # Prepare the semantic embeddings
    to_compute_semantic_embs = True
    if os.path.isfile(precomputed_semantic_embs):
        with open(precomputed_semantic_embs, "rb") as f:
            precomputed_embs_dict = pickle.load(f)
        # Check if novel classes exist in precomputed embs
        if all(x in precomputed_embs_dict.keys() for x in classes):
            return precomputed_embs_dict
    
    
    if to_compute_semantic_embs:
        # We take the average for classes e.g. "hot dog", "parking meter".
        word_embs_dict = {x: None for cls in classes for x in cls.split(" ")}
        with open(semantic_corpus, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.split("\n")[0].split(" ")
                word = line[0]
                if word in word_embs_dict:
                    emb = np.asarray([float(x) for x in line[1:]])
                    word_embs_dict[word] = emb
                if all([v is not None for k, v in word_embs_dict.items()]):
                    # Break if all words have found its embedding.
                    break
        # check all words have a corresponding semantic embeddings
        none_embs = [x for x, emb in word_embs_dict.items() if emb is None]
        if len(none_embs) > 0:
            msg = "Some classes (words) are not in the corpus and will be skipped in inference:\n"
            msg += "\n".join("  " + colored(x, "blue") for x in none_embs)
            logger.info(msg)
        # Remove none classes
        def is_valid(cls, none_embs):
            for x in cls.split(" "):
                if x in none_embs:
                    return False
            return True
        classes = [x for x in classes if is_valid(x, none_embs)]
        
        class_embs_dict = {}
        for cls in classes:
            emb = [word_embs_dict[x] for x in cls.split(" ") if word_embs_dict[x] is not None]
            emb = np.stack(emb, axis=0).mean(axis=0)
            class_embs_dict[cls] = emb

    # Save semantic embeddings to avoid repeated computations.
    if os.path.isfile(precomputed_semantic_embs):
        with open(precomputed_semantic_embs, "rb") as f:
            precomputed_embs_dict = pickle.load(f)
        precomputed_embs_dict.update(class_embs_dict)
        with open(precomputed_semantic_embs, "wb") as f:
            pickle.dump(precomputed_embs_dict, f)
    else:
        with open("./datasets/precomputed_semantic_embeddings.pkl", "wb") as f:
            pickle.dump(class_embs_dict, f)

    return class_embs_dict


class ZeroShotPredictor(nn.Module):
    """
    Zero-shot predictors for discovering objects from novel categories.
    """
    def __init__(self, cfg, known_classes, novel_classes):
        super(ZeroShotPredictor, self).__init__()
        # fmt: off
        self.cls_agnostic_bbox_reg     = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.pre_inference_thresh      = cfg.ZERO_SHOT.PRE_INFERENCE_THRESH
        self.post_inference_thresh     = cfg.ZERO_SHOT.POST_INFERENCE_THRESH
        self.topk_known_classes        = cfg.ZERO_SHOT.TOPK_KNOWN_CLASSES
        self.detections_per_image      = cfg.ZERO_SHOT.DETECTIONS_PER_IMAGE
        self.precomputed_semantic_embs = cfg.ZERO_SHOT.PRECOMPUTED_SEMANTIC_EMBEDDINGS
        self.semantic_corpus           = cfg.ZERO_SHOT.SEMANTIC_CORPUS
        # fmt: on
        self._init_embs(known_classes, novel_classes)
        
    def _init_embs(self, known_classes, novel_classes):
        """
        Initilize semantic embeddings for classes.
        """
        # laading semantic word embeddings.
        class_embs_dict = load_semantic_embeddings(
            self.semantic_corpus,
            known_classes + novel_classes,
            self.precomputed_semantic_embs,
        )
        
        assert all([x in class_embs_dict for x in known_classes])
        self.known_classes = known_classes
        self.novel_classes = [x for x in novel_classes if x in class_embs_dict]

        self.known_class_embs = torch.stack([
            torch.as_tensor(class_embs_dict[x]) for x in known_classes
        ], dim=0)

        if len(self.novel_classes) == 0:
            return
        self.novel_class_embs = torch.stack([
            torch.as_tensor(class_embs_dict[x]) for x in novel_classes if x in class_embs_dict
        ], dim=0)

    def inference(self, scores, proposal_deltas, proposals):
        """
        Args:
            scores: predicted probability of known classes.
            proposal_deltas: predicted box deltas. If `CLS_AGNOSTIC_BBOX_REG` = True, it has
                shape (N, 4), otherwise its shape is (N, C * 4), where N is the number of
                instances and C is the number of known classes.
        """
        device = scores.device
        num_novel_classes = len(self.novel_classes)
        
        num_instances = len(scores)
        if num_instances == 0 or num_novel_classes == 0:
            return scores, proposal_deltas

        known_class_embs = self.known_class_embs.to(device)
        novel_class_embs = self.novel_class_embs.to(device)
        
        novel_scores = torch.zeros(
            (num_instances, num_novel_classes), dtype=scores.dtype, device=device
        )
        # 1. For the boxes whose score of known classes is less than threshold, we perform
        # zero-shot inference to reason its score of being the given novel classes.
        known_scores = scores[:, :-1] # excluding background scores
        max_known_scores = torch.max(known_scores, dim=1)[0]
        enable = torch.nonzero(
            (max_known_scores < self.pre_inference_thresh) & (max_known_scores > 1e-3)
        ).squeeze(1)
        # 2. Obtain the scores of top K known classes.
        known_scores, kept_idxs = torch.sort(known_scores[enable], dim=-1, descending=True)
        known_scores = known_scores[:, :self.topk_known_classes]
        kept_idxs = kept_idxs[:, :self.topk_known_classes]
        
        # 3. Estimate the semantic embeddings of boxes
        base_embs = known_class_embs[kept_idxs]
        norm_factors = known_scores.sum(dim=-1, keepdim=True)
        base_wgts = known_scores / norm_factors.repeat(1, self.topk_known_classes)
        pred_embs = base_embs * base_wgts.unsqueeze(-1).repeat(1, 1, base_embs.size(-1))
        pred_embs = torch.sum(pred_embs, dim=1)

        # 4. Predict scores for novel classes by computing cosine similarity.
        emb_norms = torch.norm(pred_embs, p=2, dim=1, keepdim=True)
        pred_embs = pred_embs.div(emb_norms.expand_as(pred_embs))
        
        emb_norms = torch.norm(novel_class_embs, p=2, dim=1, keepdim=True)
        novel_class_embs = novel_class_embs.div(emb_norms.expand_as(novel_class_embs))
        
        novel_scores[enable, :] = torch.mm(
            pred_embs, novel_class_embs.permute(1, 0)
        ).to(novel_scores.dtype)
        # Reweight interactness scores
        interactness_scores = torch.sigmoid(proposals[0].interactness_logits)
        novel_scores = novel_scores * interactness_scores.unsqueeze(1).repeat(1, num_novel_classes)

        # 5. Post processing. Remove predictions whose score < post_inference_thresh.
        novel_scores[novel_scores < self.post_inference_thresh] = 0.
        novel_scores[proposals[0].is_person == 1, :] = 0.
        # Maximum number of detections to keep
        thresh = torch.topk(novel_scores.reshape(-1), self.detections_per_image)[0][-1]
        novel_scores[novel_scores <= thresh] = 0.
        
        novel_scores = torch.clamp(novel_scores * 3, min=0., max=1.)


        # Always keep the background as the last.
        scores = torch.cat([scores[:, :-1], novel_scores, scores[:, -1:]], dim=-1)
        if not self.cls_agnostic_bbox_reg:
            proposal_deltas = torch.cat([
                proposal_deltas,
                torch.zeros((num_instances, num_novel_classes * 4), device=device)
            ], dim=-1)
        return scores, proposal_deltas