import logging
import time
from typing import List, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import SwissKnifeConfig
from .gsi_softmax import GSISoftmaxGenerator
from .rrm_judge import RRMJudge

logger = logging.getLogger(__name__)


class RRMCandidatePoolGenerator:
    """Generates N response candidates in parallel using GSI, and ranks them
    using RRM-7B judge to select the tournament champion response.
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        drafter_model: PreTrainedModel,
        drafter_tokenizer: PreTrainedTokenizer,
        verifier_model: PreTrainedModel,
        verifier_tokenizer: PreTrainedTokenizer,
        rrm_model: PreTrainedModel,
        rrm_tokenizer: PreTrainedTokenizer,
        blade_model: Optional[PreTrainedModel] = None,
    ):
        self.cfg = cfg
        self.gsi_generator = GSISoftmaxGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        self.rrm_judge = RRMJudge(
            model=rrm_model,
            tokenizer=rrm_tokenizer,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        blade: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Generate N candidates in parallel, run Elo tournament via RRM-7B,
        and return (champion_response, all_candidates).
        """
        N = self.cfg.rrm_n_candidates
        logger.info("Generating %d candidate responses in parallel via GSI...", N)

        t0 = time.time()
        prompts = [prompt] * N
        candidates = self.gsi_generator.generate_batched(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
            blade=blade,
        )
        t_gen = time.time() - t0
        logger.info("Generated %d responses in %.2fs", N, t_gen)

        if verbose:
            for i, cand in enumerate(candidates):
                logger.debug("Candidate %d:\n%s", i, cand)

        t_tournament = time.time()
        champion_idx = self.rrm_judge.run_tournament(
            prompt=prompt,
            candidates=candidates,
            temperature=self.cfg.elo_temperature,
        )
        t_tourney_end = time.time() - t_tournament
        logger.info("RRM tournament resolved champion in %.2fs. Selected Candidate: %d", t_tourney_end, champion_idx)

        return candidates[champion_idx], candidates
