"""GRID prefix-mask block, isolated from the Transformer for CPU testing.

Copyright (c) 2025 Snap Inc. All rights reserved.
Distributed under the non-commercial research license in LICENSE.GRID.

Upstream: https://github.com/snap-research/GRID
Commit: 2fe3475b2d369580234093f35d52b1a2f54d0472
Source: src/models/modules/semantic_id/tiger_generation_model.py
Function: SemanticIDGenerativeRecommender._beam_search_one_step

The block is wrapped for CPU tests. The upstream indentation bug is intentionally
retained; restore-grid patches it.
"""

import torch


class PrefixMask:
    def probabilities(self, candidate_logits, generated_ids, hierarchy, batch_size):
        # pruning the beams that cannot be mapped to a valid item
        if self.should_check_prefix:
            if generated_ids is None:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.arange(
                        self.num_embeddings_per_hierarchy,
                        device=candidate_logits.device,
                    ).unsqueeze(1)
                )
                candidate_logits[:, ~valid_prefix_mask] = float("-inf")
            else:
                # we prune all beams with prefixes that cannot be mapped to a valid item
                valid_prefix_mask = self._check_valid_prefix(
                    torch.cat(
                        [
                            generated_ids.reshape(-1, hierarchy).repeat_interleave(
                                self.num_embeddings_per_hierarchy, dim=0
                            ),
                            torch.arange(
                                self.num_embeddings_per_hierarchy,
                                device=candidate_logits.device,
                            )
                            .repeat(self.top_k_for_generation * batch_size)
                            .unsqueeze(1),
                        ],
                        dim=1,
                    )
                ).reshape(-1, self.num_embeddings_per_hierarchy)
            candidate_logits[~valid_prefix_mask] = float("-inf")

        candidate_logits = torch.nn.functional.softmax(candidate_logits, dim=-1)
        return candidate_logits
