from abc import ABC, abstractclassmethod
from typing import List, Tuple, Union
from torch import Tensor
import torch
from datasets import load_metric
from rl4lms.envs.text_generation.observation import Observation
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM
from rl4lms.envs.text_generation.metric import (
    CIDERMetric,
    MeteorMetric,
    BERTScoreMetric,
    BLEUMetric,
    SpiceMetric,
    ParentToTTo,
    RougeLMax,
    TERMetric,
    chrFmetric,
    Perplexity,
    GPT2Perplexity,
    IntentAccuracyDailyDialog,
    IntentAccuracyDailyDialogNoisy,
    IntentAccuracyDailyDialogConditional,
    IntentAccuracyDailyDialogPlusDECODEMetric,
    DiffusionImageGenerationSimilarityMetric
    )
import numpy as np
from typing import List, Dict, Any


class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        """
        Callable for reward functions for text generation

        Args:
            current_observation (Observation): previous observation (s)
            action (int): action performed (a) at s
            next_observation (Observation): observation after the action was performed (s')
            done (bool): whether the episode is finished or not
            meta_info (dict) - other information regarding textual sample
        Returns:
            float: scalar reward
        """
        raise NotImplementedError


class BatchedRewardFunction(ABC):
    """
    Computes rewards for several instances at once
    """

    @abstractclassmethod
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        An abstract class for batched reward functions for text generation
        """
        raise NotImplementedError


### Automated reward functions ###########################


class CommonGenPenaltyShapingFunction(RewardFunction):
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            prompt_text = next_observation.prompt_or_input_text
            prefix = "generate a sentence with: "
            concept_n_grams = prompt_text.split(prefix)[1][:-1]

            if (
                concept_n_grams.lower() in next_observation.context_text.lower()
                or prefix in next_observation.context_text.lower()
                or "generate" in next_observation.context_text.lower()
                or "sentence" in next_observation.context_text.lower()
            ):
                penalty_score = -1
            else:
                penalty_score = 0
            return penalty_score
        return 0


class BatchedCommonGenPenaltyShapingFunction(BatchedRewardFunction):
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        scores = []
        for done, prompt_text, gen_text in zip(dones, prompt_texts, gen_texts):
            if done:
                prefix = "generate a sentence with: "
                concept_n_grams = prompt_text.split(prefix)[1][:-1]

                if (
                    concept_n_grams.lower() in gen_text.lower()
                    or prefix in gen_text.lower()
                    or "generate" in gen_text.lower()
                    or "sentence" in gen_text.lower()
                ):
                    penalty_score = -1
                else:
                    penalty_score = 0
                scores.append(penalty_score)
        return scores


class MeteorRewardFunction(RewardFunction):
    def __init__(self, shaping_fn: str = None) -> None:
        super().__init__()
        self._metric = MeteorMetric()
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute meteor at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/meteor"][1]

            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                score = score + aux_score
            return score
        return 0


class RougeRewardFunction(RewardFunction):
    def __init__(
        self, rouge_type: str, shaping_fn: str = None, use_single_ref: bool = True
    ) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self._rouge_type = rouge_type
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )
        self._use_single_ref = use_single_ref

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )
            reward = metric_results[self._rouge_type].mid.fmeasure
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0


class RougeCombined(RewardFunction):
    def __init__(self, shaping_fn: str = None) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            references = [next_observation.target_or_reference_texts[0]]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )

            rouge_keys = ["rouge1", "rouge2", "rougeL"]
            scores = [
                metric_results[rouge_type].mid.fmeasure for rouge_type in rouge_keys
            ]
            reward = np.mean(scores)
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0


class BERTScoreRewardFunction(RewardFunction):
    def __init__(self, language: str = "en") -> None:
        super().__init__()
        self._metric = BERTScoreMetric(language)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(None, predicted, references)
            bert_score = metric_results["semantic/bert_score"][1]
            return bert_score
        return 0


class BLEURewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = BLEUMetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(None, predicted, references)
            bleu_score = metric_results["lexical/bleu"][1]
            return bleu_score
        return 0


class SacreBleu(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = load_metric("sacrebleu")
        self._args = args

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(
                predictions=predicted, references=references, **self._args
            )
            return metric_results["score"] / 100
        return 0


class SpiderRewardFunction(BatchedRewardFunction):
    def __init__(
        self, spice_coeff: float, cider_coeff: float, shaping_fn: str = None
    ) -> None:
        """
        Spice + Cider
        """
        super().__init__()
        self._spice_metric = SpiceMetric()
        self._cider_metric = CIDERMetric()
        self._spice_coeff = spice_coeff
        self._cider_coeff = cider_coeff
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        prompts = []
        gens = []
        refs = []
        indices_with_done = []
        rewards = torch.zeros(len(prompt_texts))
        for ix, (prompt, gen, ref, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, dones)
        ):
            if done:
                prompts.append(prompt)
                gens.append(gen)
                refs.append(ref)
                indices_with_done.append(ix)

        if len(indices_with_done) > 0:
            spice_scores = self._spice_metric.compute(prompts, gens, refs)[
                "lexical/spice"
            ][0]
            cider_scores = self._cider_metric.compute(prompts, gens, refs)[
                "lexical/cider"
            ][0]
            total_scores = self._spice_coeff * np.array(
                spice_scores
            ) + self._cider_coeff * np.array(cider_scores)

            if self._shaping_fn is not None:
                aux_scores = self._shaping_fn(prompt_texts, gen_texts, ref_texts, dones)
            else:
                aux_scores = [0] * len(indices_with_done)

            for ind, score, aux_score in zip(
                indices_with_done, total_scores, aux_scores
            ):
                rewards[ind] = score + aux_score

        return rewards


#############################################################################

########## Learned Reward Functions##########################################


class LearnedRewardFunction(RewardFunction):
    def __init__(
        self, model_name: str, label_ix: int, include_prompt_for_eval: bool = True
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._metric_tokenizer.truncation_side = "left"
        self._metric_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._label_ix = label_ix
        self._include_prompt_for_eval = include_prompt_for_eval

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_text = (
                current_observation.prompt_or_input_text
                if self._include_prompt_for_eval
                else ""
            )
            generated_text += next_observation.context_text

            with torch.no_grad():
                encoded = self._metric_tokenizer(
                    generated_text, return_tensors="pt", truncation=True, padding=True
                )
                outputs = self._metric_model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits.flatten(), dim=0)
                score = scores[self._label_ix].item()
                return score
        return 0


class BLEURTRewardFunction(RewardFunction):
    def __init__(self, checkpoint: str = None):
        super().__init__()
        self._metric = load_metric("bleurt", checkpoint=checkpoint)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(
                predictions=predicted, references=references
            )
            score = metric_results["scores"][0]
            return score
        return 0


class PARENTRewardFunction(RewardFunction):
    """
    PARENT F1 score as the reward
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = ParentToTTo()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_texts = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, generated_texts, None, meta_infos)
            reward = scores["table_to_text/parent_overall_f_score"][0][0]
            return reward
        return 0


class RougeLMaxRewardFunction(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = RougeLMax(**args)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, predicted, references, meta_infos)
            reward = scores["lexical/rouge_l_max"][0][0]
            return reward
        return 0


class TER(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = TERMetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute score at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/ter"][1]
            score = 1 - score
            return score
        return 0


class chrF(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = chrFmetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute score at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/chrf"][1]
            return score
        return 0

class PictionaryReward(BatchedRewardFunction):
    def __init__(self):
        self._metric = None

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        if self._metric is None:
            self._metric = PictionaryMetric()
        
        scores_dict = self._metric.compute(
            prompt_texts, gen_texts, ref_texts, meta_infos, split_name="train"
        )
        scores = scores_dict["pictionary_metric"]
        batch_scores = torch.tensor(scores[0]).squeeze()      # all metrics do .tolist()
        mean_score = scores[1]
        
        print(f"Got net rewards: {rewards}, len(rewards): {len(rewards)}, mean: {np.mean(rewards)}")
        return rewards.tolist()


class  DiffusionImageGenerationSimilarityReward(BatchedRewardFunction):
    def __init__(self, shape: bool = True, use_topic_only_for_worse: bool = False, arg1: float = 0.0) -> None:
        super().__init__()
        self._use_topic_only_for_worse = use_topic_only_for_worse
        self._metric = None
        self._shape = shape
        self._ppl_coeff = 0.2
        self._shaping_metric = Perplexity(stride=32, tokenizer_id="gpt2")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"

        self._gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self._gpt2_model.to(self._device)


    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        if self._metric is None:
            self._metric = DiffusionImageGenerationSimilarityMetric(
                    use_topic_only_for_worse=self._use_topic_only_for_worse)

        # Example gen_texts (only last trajectory is finished): [' a', ' a small', ' a small courtyard', ' a small courtyard in', ' a small courtyard in the', ' a small courtyard in the center', ' a small courtyard in the center of', ' a small courtyard in the center of a', ' a small courtyard in the center of a house', ' a small courtyard in the center of a house.',
        rewards = np.zeros(len(gen_texts))

        # Commented out because we will not wait until done to give reward
        # done_prompt_texts = []
        # done_gen_texts = []
        #done_ref_texts = []
        # done_meta_infos = []
        # done_ixs = []
        # for ix, (prompt, gen, ref, meta_info, done) in enumerate(
        #     zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        #):
        #     if done:
        #         done_prompt_texts.append(prompt)
        #         done_gen_texts.append(gen)
        #         done_ref_texts.append(ref)
        #         done_meta_infos.append(meta_info)
        #         done_ixs.append(ix)
        
                 # This works but the the probability is very near 0, unclear how to map to 0 to 1
        if self._shape:
            # it will return nothing if it's train split
            # calculates it on the generated texts for an untuned gpt2 model
            # tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # for gt in range(len(gen_texts)):
            #    encodings = tokenizer(gen_texts[gt], return_tensors="pt")
            #    dev_ids = encodings.input_ids.to(self._device)
            #    outputs = self._gpt2_model(dev_ids)
            #    seq_len = encodings.input_ids.shape[-1]
            #    nlls = outputs[0]        # (1, n tokens, vocab)
            #    nlls = torch.gather(nlls, dim=-1, index=dev_ids.unsqueeze(0))  # (1, n tokens)
            #    ppl = nlls.sum() / seq_len
            # score = self._shaping_metric.compute(
            #    prompt_texts, gen_texts, gen_texts,
            #    meta_infos, model=self._gpt2_model, split_name="val"
            #)
            #ppl_score = torch.exp(-score["fluency_metrics/gpt2_ppl"][1])
            #rint(f"ppl_score: {ppl_score}")
            ppl_score = self._shaping_metric.compute(prompt_texts, gen_texts, 
                gen_texts, None, self._gpt2_model)
           
            # it yells at you saying it's too long but still computes a value
            # and it shouldn't be too long :(; returns a tuple with None is 0th position
            ppl = ppl_score["fluency_metrics/perplexity"][1]
            # 10 PPL = 0.13 reward, 20 = 0.0183
            # scaled_ppl = self._ppl_coeff * np.exp(-0.20*ppl)
            scaled_ppl = -ppl / 50.
            print(f"Unscaled batch PPL was: {ppl}, Scaled PPL reward was: {scaled_ppl}")
            rewards += scaled_ppl

        # scores_dict = self._metric.compute(
        #   done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
        #)
        scores_dict = self._metric.compute(
            prompt_texts, gen_texts, ref_texts, meta_infos, split_name="train"
        )
        scores = scores_dict["diffusion_image_similarity_score"]
        batch_scores = torch.tensor(scores[0]).squeeze()      # for some reason, all metrics do .tolist()
        mean_score = scores[1]
        try:
            # rewards[done_ixs] +=  (1 - self._ppl_coeff) * np.array(batch_scores)
            # every "trajectory" adds 1 token per step; give rewards for all not just done ones
            scaled_scores = (1 - self._ppl_coeff) * np.array(batch_scores)
            assert scaled_scores.shape == batch_scores.shape
            rewards +=  scaled_scores
        except Exception as e: 
            print(f"Had error in reward function! {e}")
            import pdb;pdb.set_trace()
        print(f"Got net rewards: {rewards}, len(rewards): {len(rewards)}, mean: {np.mean(rewards)}")
        return rewards.tolist()


class IntentAccuracy(BatchedRewardFunction):
    def __init__(
        self, shape: bool = True, intent_coeff: float = 1.0, auto_coeff: float = 1.0
    ) -> None:
        super().__init__()
        self._metric = None
        self._shape = shape
        self._intent_coeff = intent_coeff
        self._auto_coeff = auto_coeff
        self._shaping_metric = MeteorMetric()

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        if self._metric is None:
            self._metric = IntentAccuracyDailyDialog()

        # compute rewards for finished episodes only
        rewards = np.zeros(len(gen_texts))

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)

                if self._shape:
                    score = self._shaping_metric.compute(
                        done_prompt_texts, done_gen_texts, done_ref_texts
                    )
                    rewards[ix] = self._auto_coeff * score["lexical/meteor"][1]

        scores = self._metric.compute(
            done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
        )["intent/accuracy"][0]
        rewards[done_ixs] += self._intent_coeff * np.array(scores)
        return rewards.tolist()

class IntentAccuracyNoisy(BatchedRewardFunction):
    def __init__(
            self, shape: bool = True, intent_coeff: float = 1.0, 
            auto_coeff: float = 1.0, pct_noise: float=0.0, noise_factor: float=1.0
    ) -> None:
        super().__init__()
        self._metric = None
        self._shape = shape
        self._intent_coeff = intent_coeff
        self._auto_coeff = auto_coeff
        self._shaping_metric = MeteorMetric()
        self._pct_noise = pct_noise
        self._noise_factor = noise_factor

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        if self._metric is None:
            self._metric = IntentAccuracyDailyDialogNoisy(pct_noise=self._pct_noise,
                    noise_factor=self._noise_factor)

        # compute rewards for finished episodes only
        rewards = np.zeros(len(gen_texts))

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)

                if self._shape:
                    score = self._shaping_metric.compute(
                        done_prompt_texts, done_gen_texts, done_ref_texts
                    )
                    rewards[ix] = self._auto_coeff * score["lexical/meteor"][1]

        scores = self._metric.compute(
            done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
        )["intent/noisy_accuracy"][0]
        rewards[done_ixs] += self._intent_coeff * np.array(scores)
        return rewards.tolist()

class IntentAccuracyConditional(BatchedRewardFunction):
    def __init__(
            self, shape: bool = True, intent_coeff: float = 1.0,
            auto_coeff: float = 1.0, min_reward: float=0.0, max_prob_threshold: float=0.
    ) -> None:
        super().__init__()
        self._metric = None
        self._shape = shape
        self._intent_coeff = intent_coeff
        self._auto_coeff = auto_coeff
        self._shaping_metric = MeteorMetric()
        self._min_reward = min_reward
        self._max_prob_threshold = max_prob_threshold

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        if self._metric is None:
            self._metric = IntentAccuracyDailyDialogConditional(min_reward=self._min_reward,
                    max_prob_threshold=self._max_prob_threshold)

        # compute rewards for finished episodes only
        rewards = np.zeros(len(gen_texts))

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)

                if self._shape:
                    score = self._shaping_metric.compute(
                        done_prompt_texts, done_gen_texts, done_ref_texts
                    )
                    rewards[ix] = self._auto_coeff * score["lexical/meteor"][1]

        scores = self._metric.compute(
            done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
        )["intent/conditional_accuracy"][0]
        rewards[done_ixs] += self._intent_coeff * np.array(scores)
        return rewards.tolist()


class IntentAccuracyPlusDECODEReward(BatchedRewardFunction):
    def __init__(
            self, shape: bool = True, intent_coeff: float = 1.0,
            auto_coeff: float = 1.0, decode_weight: float = 0.0
    ) -> None:
        super().__init__()
        self._metric = None
        self._shape = shape
        self._intent_coeff = intent_coeff
        self._auto_coeff = auto_coeff
        self._shaping_metric = MeteorMetric()
        self._decode_weight = decode_weight

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        if self._metric is None:
            self._metric = IntentAccuracyDailyDialogPlusDECODEMetric(
                    decode_weight=self._decode_weight)

        # compute rewards for finished episodes only
        rewards = np.zeros(len(gen_texts))

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)

                if self._shape:
                    score = self._shaping_metric.compute(
                        done_prompt_texts, done_gen_texts, done_ref_texts
                    )
                    rewards[ix] = self._auto_coeff * score["lexical/meteor"][1]

        scores = self._metric.compute(
            done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
        )["intent/intent_plus_decode_metric"][0]
        rewards[done_ixs] += self._intent_coeff * np.array(scores)
        return rewards.tolist()




if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    predictions = ["hello there general kenobi"]
    references = [["hello there general kenobi", "hello there!!"]]
    observation = Observation(
        None, None, None, None, None, predictions, references, None, None, None, None
    )

    reward_fn = DiffusionImageGenerationSimilarityReward()
    # reward_fn = IntentAccuracyPlusDECODEReward(decode_weight=0.5)
    prompt_text = ["kenobi star wars"]
    meta_infos = [{"intent":[2]}]


    outputs = reward_fn(prompt_text, predictions, references, [True], meta_infos)
    print(f"Got output of reward function: {outputs}")

    # reward_fn = MeteorRewardFunction()
    # print(reward_fn(None, None, observation, True))

    # reward_fn = chrF()
    # print(reward_fn(None, None, observation, True))

    # reward_fn = RougeCombined()
    # print(reward_fn(None, None, observation, True))

    # reward_fn = RougeRewardFunction(rouge_type="rouge1")
    # print(reward_fn(None, None, observation, True))

    # reward_fn = RougeRewardFunction(rouge_type="rouge2")
    # print(reward_fn(None, None, observation, True))

    # reward_fn = RougeRewardFunction(rouge_type="rougeL")
    # print(reward_fn(None, None, observation, True))

    # reward_fn = BERTScoreRewardFunction(language="en")
    # print(reward_fn(None, None, observation, True))

    # reward_fn = BLEURewardFunction()
    # print(reward_fn(None, None, observation, True))

    # reward_fn = BLEURTRewardFunction()
    # print(reward_fn(None, None, observation, True))
