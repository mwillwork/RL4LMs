import os
from PIL import Image, PngImagePlugin
import json
from datetime import datetime as dt
from typing import List, Tuple, Union
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from abc import abstractmethod
import numpy as np
from datasets import load_metric
from gem_metrics.msttr import MSTTR
from gem_metrics.ngrams import NGramStats
from rl4lms.envs.text_generation.caption_metrics.cider import Cider
from rl4lms.envs.text_generation.caption_metrics.spice.spice import Spice
from gem_metrics.texts import Predictions
from rl4lms.envs.text_generation.summ_metrics.summa_c import SummaCConv, SummaCZS
from rl4lms.data_pools.task_utils.totto.eval_utils import compute_parent, compute_bleu
from rl4lms.data_pools.custom_text_generation_pools import DailyDialog
from tqdm import tqdm
import copy
import rouge
import logging

logging.getLogger('PIL').setLevel(logging.INFO)
# PngImagePlugin.DEBUG = False

class BaseMetric:
    @abstractmethod
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        """
        Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores (if any) and corpus level score

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        """
        raise NotImplementedError


class LearnedRewardMetric(BaseMetric):
    def __init__(
        self,
        model_name: str,
        label_ix: int,
        batch_size: int,
        include_prompt_for_eval: bool = True,
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.truncation_side = "left"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self._device
        )
        self._label_ix = label_ix
        self._batch_size = batch_size
        self._include_prompt_for_eval = include_prompt_for_eval

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Dict[str, float]:
        all_scores = []
        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_gen_texts = generated_texts[
                current_ix : current_ix + self._batch_size
            ]
            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]

            if self._include_prompt_for_eval:
                batch_gen_texts = [
                    (prompt + gen)
                    for gen, prompt in zip(batch_gen_texts, batch_prompt_texts)
                ]
            encoded = self._tokenizer(
                batch_gen_texts, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores[:, self._label_ix].tolist()
                all_scores.extend(scores)
            current_ix += self._batch_size

        metric_dict = {
            "semantic/learned_automodel_metric": (all_scores, np.mean(all_scores))
        }
        return metric_dict


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("meteor")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):

        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]

        metric_dict = {"lexical/meteor": (None, score)}
        return metric_dict


class RougeMetric(BaseMetric):
    def __init__(self, use_single_ref: bool = True) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self._use_single_ref = use_single_ref

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        if self._use_single_ref:
            # TBD: this is required for CNN/DM dataset, without this we get low scores
            # TBD: needs investigation
            ref_texts = [ref[0] for ref in reference_texts]
        else:
            ref_texts = reference_texts

        metric_results = self._metric.compute(
            predictions=generated_texts, references=ref_texts, use_stemmer=True
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"lexical/rouge_{rouge_type}"] = (None, rouge_score)
        return metric_dict


class BERTScoreMetric(BaseMetric):
    def __init__(self, language: str) -> None:
        super().__init__()
        self._metric = load_metric("bertscore")
        self._language = language
        # since models are loaded heavily on cuda:0, use the last one to avoid memory
        self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        with torch.no_grad():
            metric_results = self._metric.compute(
                predictions=generated_texts,
                references=reference_texts,
                lang=self._language,
                device=self._last_gpu,
            )
            bert_scores = metric_results["f1"]
            corpus_level_score = np.mean(bert_scores)
            metric_dict = {"semantic/bert_score": (bert_scores, corpus_level_score)}
            return metric_dict


class BLEUMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("bleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        tokenized_predictions = []
        tokenized_reference_texts = []
        for prediction, refs in zip(generated_texts, reference_texts):
            tokenized_prediction = prediction.split()
            tokenized_refs = [ref.split() for ref in refs]
            tokenized_predictions.append(tokenized_prediction)
            tokenized_reference_texts.append(tokenized_refs)

        try:
            metric_results = self._metric.compute(
                predictions=tokenized_predictions, references=tokenized_reference_texts
            )
            bleu_score = metric_results["bleu"]
            metric_dict = {"lexical/bleu": (None, bleu_score)}
            return metric_dict
        except Exception as e:
            return {"lexical/bleu": (None, "n/a")}


class BLEURTMetric(BaseMetric):
    def __init__(self, config_name: str = None) -> None:
        super().__init__()
        self._metric = load_metric("bleurt", config_name=config_name)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"semantic/bleurt": (metric_results["scores"], corpus_score)}
        return metric_dict


def get_generated_and_predictions(
    prompt_texts: List[str],
    generated_texts: List[str],
    reference_texts: List[List[str]],
    split_name: str,
):
    split_name = "" if split_name is None else split_name
    preds = {}
    refs = {}
    for ix, (prompt_text, gen_text, ref_text) in enumerate(
        zip(prompt_texts, generated_texts, reference_texts)
    ):
        preds[split_name + prompt_text] = [gen_text]
        refs[split_name + prompt_text] = ref_text
    return preds, refs


def get_individual_scores(
    prompt_texts: List[str], split_name: str, scores_dict: Dict[str, float]
):
    split_name = "" if split_name is None else split_name
    scores = []
    for prompt_text in prompt_texts:
        scores.append(scores_dict.get(split_name + prompt_text, "n/a"))
    return scores


class CIDERMetric(BaseMetric):
    def __init__(self) -> None:
        self._metric = Cider()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions, references = get_generated_and_predictions(
            prompt_texts, generated_texts, reference_texts, split_name
        )
        (
            corpus_score,
            individual_scores,
        ) = self._metric.compute_score(references, predictions)
        individual_scores = get_individual_scores(
            prompt_texts, split_name, individual_scores
        )

        metric_dict = {"lexical/cider": (individual_scores, corpus_score)}
        return metric_dict


class SpiceMetric(BaseMetric):
    def __init__(self) -> None:
        self._metric = Spice()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions, references = get_generated_and_predictions(
            prompt_texts, generated_texts, reference_texts, split_name
        )
        (
            corpus_score,
            individual_scores,
        ) = self._metric.compute_score(references, predictions)

        individual_scores = get_individual_scores(
            prompt_texts, split_name, individual_scores
        )

        metric_dict = {"lexical/spice": (individual_scores, corpus_score)}
        return metric_dict


class DiversityMetrics(BaseMetric):
    def __init__(self, window_size: int = 100) -> None:
        self._msttr_metric = MSTTR(window_size=window_size)
        self._n_gram_metric = NGramStats()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        predictions = Predictions(data={"filename": "", "values": generated_texts})
        diversity_metrics = {}
        msttr_metrics = self._msttr_metric.compute(None, predictions)
        n_gram_metrics = self._n_gram_metric.compute(None, predictions)

        for key, value in msttr_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)
        for key, value in n_gram_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)

        return diversity_metrics


class SummaCZSMetric(BaseMetric):
    """
    Consistency metric for summarization

    https://github.com/tingofurro/summac/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._scorer = SummaCZS(**kwargs)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._scorer.score(prompt_texts, generated_texts)
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"consistency/summaczs": (metric_results["scores"], corpus_score)}
        return metric_dict


class SummaCConvMetric(BaseMetric):
    """
    Consistency metric for summarization

    https://github.com/tingofurro/summac/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._scorer = SummaCConv(**kwargs)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._scorer.score(prompt_texts, generated_texts)
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {
            "consistency/summacconv": (metric_results["scores"], corpus_score)
        }
        return metric_dict


class Perplexity(BaseMetric):
    def __init__(
        self,
        stride: int,
        tokenizer_id: str,
        model_type: str = "causal",
        use_text_from_meta_data: bool = False,
    ) -> None:
        super().__init__()
        self._tokenizer_id = tokenizer_id
        self._model_type = model_type
        self._stride = stride
        self._use_text_from_meta_data = use_text_from_meta_data

    def get_device(self, model: PreTrainedModel):
        try:
            return model.transformer.first_device
        except:
            return model.device

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        if split_name == "train":
            return {}

        if self._model_type != "causal":
            raise NotImplementedError

        # we compute perplexity on reference texts
        if self._use_text_from_meta_data:
            reference_texts = [info["reference"] for info in meta_infos]
        else:
            reference_texts = [ref for refs in reference_texts for ref in refs]
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
        encodings = tokenizer("\n\n".join(reference_texts), return_tensors="pt")

        device = self.get_device(model)

        nlls = []
        max_length = model.config.n_positions
        for i in tqdm(range(0, encodings.input_ids.size(1), self._stride)):
            begin_loc = max(i + self._stride - max_length, 0)
            end_loc = min(i + self._stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # run on last device
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return {
            "fluency_metrics/perplexity": (
                None,
                torch.exp(torch.stack(nlls).sum() / end_loc).item(),
            )
        }


class ParentToTTo:
    """
    Official version
    """

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]],
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        tables = [info["raw_table"] for info in meta_infos]
        parent_overall, parent_overlap, parent_non_overlap = compute_parent(
            generated_texts, tables
        )

        metric_results = {}
        metric_names = ["parent_overall", "parent_overlap", "parent_non_overlap"]
        metric_values = [parent_overall, parent_overlap, parent_non_overlap]
        for name, value in zip(metric_names, metric_values):
            metric_results[f"table_to_text/{name}/precision"] = (
                None,
                value["precision"],
            )
            metric_results[f"table_to_text/{name}/recall"] = (None, value["recall"])

            # individual f-scores - fetch only for overall since we don't know for which samples
            if name == "parent_overall":
                f_scores = value["all_f"]
            else:
                f_scores = None

            metric_results[f"table_to_text/{name}_f_score"] = (
                f_scores,
                value["f_score"],
            )
        return metric_results


class BLEUToTTo:
    """
    Official version
    """

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]],
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        tables = [info["raw_table"] for info in meta_infos]
        bleu_overall, bleu_overlap, bleu_non_overlap = compute_bleu(
            generated_texts, tables
        )

        metric_results = {
            "table_to_text/bleu_overall": (None, bleu_overall),
            "table_to_text/bleu_overlap": (None, bleu_overlap),
            "table_to_text/bleu_non_overlap": (None, bleu_non_overlap),
        }
        return metric_results


class RougeLMax(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = rouge.Rouge(metrics=["rouge-l"], **args)

    def _rouge_max_over_ground_truths(self, prediction, ground_truths):
        """
        Computes max of Rouge-L (https://github.com/allenai/unifiedqa/blob/bad6ef339db6286f0d8bd0661a2daeeb0f800f59/evaluation/evaluate_narrativeqa.py#L25)
        """
        # load stemmer
        self._metric.load_stemmer(self._metric.ensure_compatibility)

        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = self._metric.get_scores(prediction, [ground_truth])
            scores_for_ground_truths.append(score)
        max_score = copy.deepcopy(score)
        max_score = max([score["rouge-l"]["f"] for score in scores_for_ground_truths])
        return max_score

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        all_scores = []
        for gen_text, ref_texts in zip(generated_texts, reference_texts):
            rouge_max_score = self._rouge_max_over_ground_truths(gen_text, ref_texts)
            all_scores.append(rouge_max_score)

        metric_dict = {"lexical/rouge_l_max": (all_scores, np.mean(all_scores))}
        return metric_dict


class SacreBLEUMetric(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._args = args
        self._metric = load_metric("sacrebleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts, **self._args
        )
        bleu_score = metric_results["score"] / 100
        metric_dict = {"lexical/sacrebleu": (None, bleu_score)}
        return metric_dict


class TERMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("ter")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/ter": (None, score)}
        return metric_dict


class chrFmetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("chrf")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/chrf": (None, score)}
        return metric_dict


class IntentAccuracyDailyDialog(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"
        self._model = self._model.to(self._device)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        def get_input_for_classifier(prompt, generated_text):
            history = prompt.split(DailyDialog.EOU_TOKEN)
            history = [utt for utt in history if utt != ""]
            last_utterance = history[-1]
            input_text = last_utterance + generated_text
            return input_text

        # we have to extract the history utterances
        input_texts = [
            get_input_for_classifier(prompt, gen)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]

        # extract target intents
        target_intents = [info["intent"][0] - 1 for info in meta_infos]

        # tokenize
        encoded = self._tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            pred_labels = torch.argmax(outputs.logits, dim=1).tolist()

        matching_scores = (np.array(pred_labels) == np.array(target_intents)).astype(
            np.int32
        )
        intent_accuracy = np.mean(matching_scores)

        metric_dict = {"intent/accuracy": (matching_scores.tolist(), intent_accuracy)}
        return metric_dict


class IntentAccuracyDailyDialogNoisy(BaseMetric):
    def __init__(self, pct_noise=0, noise_factor=1) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"
        self._model = self._model.to(self._device)
        self._pct_noise = pct_noise
        self._noise_factor = noise_factor
        print(f"Setting pct noise to: {self._pct_noise}")
        
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        def get_input_for_classifier(prompt, generated_text):
            history = prompt.split(DailyDialog.EOU_TOKEN)
            history = [utt for utt in history if utt != ""]
            last_utterance = history[-1]
            input_text = last_utterance + generated_text
            return input_text

        # print(f"Metric compute prompt_texts: {prompt_texts}, generated_texts: {generated_texts}")

        # we have to extract the history utterances
        input_texts = [
            get_input_for_classifier(prompt, gen)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]

        # extract target intents
        target_intents = [info["intent"][0] - 1 for info in meta_infos]

        # tokenize
        encoded = self._tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            
            if np.random.sample() < self._pct_noise:
                print(f"Noising reward scores - pct noise: {self._pct_noise}, noise_factor: {self._noise_factor}")
                logits_std = torch.std(outputs.logits)
                sz = outputs.logits.size()
                d = outputs.logits.device
                noise_tensor = torch.normal(torch.zeros(size=sz).to(d), logits_std*self._noise_factor).to(d)
                new_logits = outputs.logits + noise_tensor
                # print(f"old logits: {outputs.logits}, noise_tensor: {noise_tensor}, new_logits: {new_logits}")
                outputs.logits = new_logits

                # probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                # NOISE_MEAN = 0.1
                # NOISE_STD = 0.1
                # noise_tensor = torch.normal(NOISE_MEAN*torch.ones(probs.size()), 
                #                            NOISE_STD*torch.ones(probs.size())).to(probs.device)
                # Will no longer add up to one but we just want the argmax of the logits below
                # new_probs = probs + noise_tensor
                # new_probs =  torch.clamp(new_probs, 0.01, 0.99)
                # new_logits = torch.log(new_probs)
                # print(f"old logits: {outputs.logits}, old probs: {probs}, noise_tensor: {noise_tensor}, new probs: {new_probs}, new_logits: {new_logits}")
                # outputs.logits = torch.log(new_probs)
            pred_labels = torch.argmax(outputs.logits, dim=1).tolist()

        matching_scores = (np.array(pred_labels) == np.array(target_intents)).astype(
            np.int32
        )
        intent_accuracy = np.mean(matching_scores)

        metric_dict = {"intent/noisy_accuracy": (matching_scores.tolist(), intent_accuracy)}
        return metric_dict


class IntentAccuracyDailyDialogConditional(BaseMetric):
    def __init__(self, min_reward=0., max_prob_threshold=0.) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"
        self._model = self._model.to(self._device)
        self._min_reward = min_reward
        self._max_prob_threshold = max_prob_threshold
        print(f"Setting min_reward to: {self._min_reward}, max_prob threshold: {max_prob_threshold}")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        def get_input_for_classifier(prompt, generated_text):
            history = prompt.split(DailyDialog.EOU_TOKEN)
            history = [utt for utt in history if utt != ""]
            last_utterance = history[-1]
            input_text = last_utterance + generated_text
            return input_text

        # print(f"Metric compute prompt_texts: {prompt_texts}, generated_texts: {generated_texts}")

        # we have to extract the history utterances
        input_texts = [
            get_input_for_classifier(prompt, gen)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]

        # extract target intents
        target_intents = [info["intent"][0] - 1 for info in meta_infos]

        # tokenize
        encoded = self._tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True
        )
        max_prob = 0.
        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )

            print(f"Calculating max prob - min reward: { self._min_reward}, max_prob threshold: {self._max_prob_threshold}")
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            max_prob = torch.max(probs, dim=1).values
            pred_labels = torch.argmax(outputs.logits, dim=1).tolist()

        # pred_labels are the classifier's predicted intents
        # target_intents are the GPT2 model's generated intents?! NO.
        matching_scores = (np.array(pred_labels) == np.array(target_intents)).astype(np.int32)

        # Only give the reward when the classifier is certain
        # Otherwise give a lower reward
        matching_above_threshold = matching_scores * (max_prob.cpu().numpy() > self._max_prob_threshold)
        matching_below_threshold = matching_scores * (max_prob.cpu().numpy() <= self._max_prob_threshold)
        adjusted_below_threshold = matching_below_threshold * self._min_reward
        matching_scores = matching_above_threshold + adjusted_below_threshold
        
        intent_accuracy = np.mean(matching_scores)

        metric_dict = {"intent/conditional_accuracy": (matching_scores.tolist(), intent_accuracy)}
        return metric_dict


class IntentAccuracyDailyDialogPlusDECODEMetric(BaseMetric):
    def __init__(self, decode_weight=0.) -> None:
        super().__init__()
        
        # THIS CLASS SHOULD ONLY BE USED INSIDE A REWARD FN NOT AS A STANDALONE METRIC
        print("DO NOT USE THIS AS A STANDALONE EVAL METRIC. IT IS JUST A REWARD FN METRIC.")


        self._tokenizer = AutoTokenizer.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )

        self._tokenizer_decode = AutoTokenizer.from_pretrained("roberta-large")

        self._model = AutoModelForSequenceClassification.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )

        self._decode_model = AutoModelForSequenceClassification.from_pretrained(
                '/home/ubuntu/roberta_classifier/test_trainer_5e-06_continue/checkpoint-400/',
                num_labels=2
        )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"
        self._model = self._model.to(self._device)
        self._decode_model = self._decode_model.to(self._device)

        self._decode_weight = decode_weight
        print(f"Setting decode_weight to: {self._decode_weight} in ensemble classifier")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        def get_input_for_classifier(prompt, generated_text, for_decode=False):
            history = prompt.split(DailyDialog.EOU_TOKEN)
            history = [utt for utt in history if utt != ""]
            last_utterance = history[-1]
            input_text = last_utterance + generated_text
            if for_decode:
                # decode input text does not expect all these <EOU> tokens
                input_text = last_utterance + "\n" + generated_text
                input_text = input_text.replace(DailyDialog.EOU_TOKEN, "\n")
                input_text = input_text.replace("<EOU", "") # Weird character, seems to appear a lot
                input_text = input_text.replace("<E", "")
            return input_text

        # print(f"Metric compute prompt_texts: {prompt_texts}, generated_texts: {generated_texts}")

        # we have to extract the history utterances
        input_texts = [
            get_input_for_classifier(prompt, gen)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]
        print(f"input_text normal: {input_texts[:200]}")

        input_texts_decode = [
            get_input_for_classifier(prompt, gen, for_decode=True)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]
        print(f"input_texts_decode: {input_texts_decode[:200]}")

        # extract target intents
        target_intents = [info["intent"][0] - 1 for info in meta_infos]


        # tokenize
        encoded = self._tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True, max_length=128
        )

        encoded_decode = self._tokenizer_decode(
            input_texts_decode, return_tensors="pt", truncation=True, padding=True, max_length=128
        )

        decode_non_contradiction_prob = 0.
        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            pred_labels = torch.argmax(outputs.logits, dim=1).tolist()

            decode_outputs = self._decode_model(
                input_ids=encoded_decode.input_ids.to(self._device),
                attention_mask=encoded_decode.attention_mask.to(self._device),
            )
            decode_probs = torch.nn.functional.softmax(decode_outputs.logits, dim=1)
            decode_non_contradiction_prob = torch.gather(decode_probs, dim=-1, index=torch.ones((decode_probs.shape[0], 1), dtype=torch.int64).to(self._device))

        
        decode_non_contradiction_prob =decode_non_contradiction_prob.cpu().numpy()
        matching_scores = (np.array(pred_labels) == np.array(target_intents)).astype(np.int32)

        # intent_accuracy = np.mean(matching_scores)

        # TODO: issues 
        # 1) what are these target intents?! should only use reward classifier predictions, 
        # 2) tokenization/context of daily dialog: could DECODE classifier handle the format

        # THIS SHOULD ONLY BE USED INSIDE A REWARD FN NOT AS A STANDALONE METRIC
        scores_term_1 = self._decode_weight * decode_non_contradiction_prob
        scores_term_2 =  (1 - self._decode_weight) * matching_scores
        final_matching_scores = scores_term_1.flatten() + scores_term_2
        final_metric = np.mean(decode_non_contradiction_prob)        
       
        print(f"Batch non-contradiction prob mean: {np.mean(decode_non_contradiction_prob)}")
        # I end up with a list of lists, so squeeze it
        # final_matching_scores = final_matching_scores.squeeze(-1)
        metric_dict = {"intent/intent_plus_decode_metric": (final_matching_scores.tolist(), final_metric)}
        return metric_dict

class PictionaryMetric(BaseMetric):
    def __init__(self, use_bert : bool = True) -> None:
        super().__init__()
        print("Creating new PictionaryMetric (and new diffusion models!)")
        from diffusers import StableDiffusionPipeline
        from transformers import ConvNextFeatureExtractor
        from transformers import ConvNextForImageClassification

        from transformers import BertTokenizer, BertModel
        # self._use_bert = use_bert
        self._use_bert = True # TODO: HACKCKCKCKC!!
        self._penalize_repetition = False
        print(f"PENALIZE REPETITION IS: {self._penalize_repetition}")

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # create the diffusion model
        diffusion_model_name = "stabilityai/stable-diffusion-2"
        self._diffusion_model = StableDiffusionPipeline.from_pretrained(
            diffusion_model_name, torch_dtype=torch.float16)
        self._diffusion_model.to("cuda")

        # create the classifier model
        classifier_model_name = "facebook/convnext-base-224"
        self._classifier_feature_extractor = ConvNextFeatureExtractor.from_pretrained(
            classifier_model_name)
        # TODO: get the list of classifier labels
        self._classifier_model = ConvNextForImageClassification.from_pretrained(
            classifier_model_name)

    # def __get_labels(self):
        # CAPTIONS_TOPICS_FILE = "/home/ubuntu/RL4LMs/rl4lms/data_pools/captions_cs224r_exp_1.jsonl"
        # all_labels = []
        # with open(CAPTIONS_TOPICS_FILE) as f:
        #    lines  = f.readlines()
        #    for idx, ln in enumerate(lines):
        #        json_data = json.loads(ln)
        #        topic = json_data["topic"]
        #        all_labels.append(topic)
        #print(f"Found {len(all_labels)} labels.")
        # all_labels = ["dog", "tree", "house", "lightning", "chair", "strawberry", "sunglasses", "stroller", "lizard", "hat", "garden", "dolphin", "hot air balloon", "moon", "boat"]
        # return all_labels

    def score_images(self, images, labels):
        """
        images:
        labels: single text label for each image
        """
        assert len(images) == len(labels)
        # for l in labels:
        #    assert l in self._classifier_labels, f"{l} was not in classifier labels"

        inputs = self._classifier_feature_extractor(images, return_tensors='pt')
        outputs = self._classifier_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits)
        label_indices = torch.tensor([int(self._classifier_model.config.label2id[l]) for l in labels])
        predictions = torch.gather(predictions, index=label_indices.unsqueeze(0), dim=-1)
        return predictions

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        if self._use_bert:
            print(f"PictionaryMetric: using BERT! Gen texts: {len(generated_texts)}")
            with torch.no_grad():
                gen_encodings = self.bert_tokenizer(generated_texts, return_tensors="pt", padding=True)
                bert_gen_embeddings = self.bert_model(gen_encodings.input_ids)

                topics = [mi["topic"] for mi in meta_infos]
                topic_encodings = self.bert_tokenizer(topics, return_tensors="pt", padding=True)
                bert_topic_embeddings = self.bert_model(topic_encodings.input_ids)

                # https://github.com/huggingface/transformers/issues/7540
                # Suggests using the last_hidden_state of the [CLS] token for each item of the batch
                # Or average all the last_hidden_state across the sequence (not using the pooler)
                # last_hidden_state.shape = (bs, sequence_length, embed_size)
                cls_embeddings_gen =  bert_gen_embeddings.last_hidden_state[:, 0, :]
                cls_embeddings_topic =  bert_topic_embeddings.last_hidden_state[:, 0, :]
                sscores = F.cosine_similarity(cls_embeddings_gen, cls_embeddings_topic)

                N = len(topics)
                generated_topics = [topics[i] in generated_texts[i] for i in range(N)]
                if np.sum(generated_topics) > 0 and self._penalize_repetition:
                    print(f"GENERATED TOPIC! Setting reward to 0: {sscores}")
                    sscores = np.where(generated_topics, 0, sscores.detach().numpy())
                    sscores = torch.tensor(sscores)

                mean_sscore = torch.mean(sscores)
                print(f"generated_texts: {generated_texts}, topics: {topics}, sscores: {sscores}, mean: {mean_sscore}")

                metric_dict = {"pictionary_metric": (sscores.tolist(), mean_sscore.item())}
                return metric_dict


        METRIC_BS = 6
        with torch.no_grad():
            print(f"PictionaryMetric: {len(generated_texts)} images in batches of {METRIC_BS}")
            all_class_scores = []
            for i in range(0, len(generated_texts), METRIC_BS):
                prompt_texts_chunk = prompt_texts[i:i+METRIC_BS]
                generated_texts_chunk = generated_texts[i:i+METRIC_BS]
                meta_infos_chunk = meta_infos[i:i+METRIC_BS]
                actual_bs = min(METRIC_BS, len(prompt_texts_chunk))
                topics_chunk = [mi["topic"] for mi in meta_infos_chunk]
                imagenet_labels_chunk = [mi["imagenet_full_label"] for mi in meta_infos_chunk]
                generated_images = self._diffusion_model(generated_texts_chunk).images
                batch_class_scores = self.score_images(generated_images, imagenet_labels_chunk)
                generated_topics = [topics_chunk[i] in generated_texts_chunk[i] for i in range(actual_bs)]
                if np.sum(generated_topics) > 0 and self._penalize_repetition:
                    batch_class_scores = np.where(generated_topics, 0, 
                            batch_class_scores.detach().numpy())
                    batch_class_scores = torch.tensor(batch_class_scores)
                    print(f"GENERATED TOPIC! Setting reward to 0: {batch_class_scores}")
                
                all_class_scores.append(batch_class_scores)
                print(f"generated_texts_chunk: {generated_texts_chunk}, topics_chunk: {topics_chunk}, class_scores: {batch_class_scores}")
            all_class_scores = torch.concat(all_class_scores, dim=-1)
            mean_score = torch.mean(all_class_scores)
            metric_dict = {"pictionary_metric": (all_class_scores.squeeze().tolist(), 
                mean_score.item())}
            return metric_dict


class DiffusionImageGenerationSimilarityMetric(BaseMetric):
    def __init__(self, use_topic_only_for_worse: bool = False) -> None:
        super().__init__()

        self._use_topic_only_for_worse = use_topic_only_for_worse
        print("Creating new DiffusionImageGenerationSimilarityMetric (and new diffusion models!)")
        print(f"Diffusion[..]Metric got use_topic_only_for_worse: {self._use_topic_only_for_worse}")
        from diffusers import StableDiffusionPipeline
        from transformers import CLIPProcessor, CLIPModel
        from torchmetrics.multimodal import CLIPScore

        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_scorer = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16", kwargs={"truncation": True})
    
        # better_model_name = "stabilityai/stable-diffusion-2"
        worse_model_name = "CompVis/stable-diffusion-v1-4"

        # self._better_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
        #    better_model_name, torch_dtype=torch.float16)
        self._worse_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
            worse_model_name, torch_dtype=torch.float16)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"

        self._clip_model.to(self._device)
        
        # self._better_diffusion_pipeline.to(self._device)
        self._worse_diffusion_pipeline.to(self._device)

    def _clip_score_image_image(self,
                                images1: Union[Tensor, List[Tensor]],
                                images2: Union[Tensor, List[Tensor]],
                                ) -> Tuple[Tensor, float]:
        """
        Doesn't do text-image similarity, does image-image similarity
        :param images1:
        :param images2:
        :return:
        """
        
        proc_images1 = self._clip_processor(text=None, images=images1, return_tensors="pt",  padding=True)
        proc_images2 = self._clip_processor(text=None, images=images2, return_tensors="pt",  padding=True)

        img_features1 = self._clip_model.get_image_features(proc_images1["pixel_values"].to(self._device))
        img_features1 = img_features1 / img_features1.norm(p=2, dim=-1, keepdim=True)

        img_features2 = self._clip_model.get_image_features(proc_images2["pixel_values"].to(self._device))
        img_features2 = img_features2 / img_features2.norm(p=2, dim=-1, keepdim=True)
        final_scores = F.cosine_similarity(img_features1, img_features2)

        return final_scores

    def normalize_rewards(self, r: torch.Tensor) -> torch.Tensor:
        r_new = torch.nn.functional.relu(r - 0.5) / 0.5
        return torch.pow(r_new, 1.5)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        METRIC_BS = 25
        with torch.no_grad():
            print(f"Diffusion[..]Metric About to generate: {len(generated_texts)} images in batches of {METRIC_BS}")
            image_similarity_scores = torch.zeros((len(generated_texts), )) 
            HACK_SKIP_IMAGE_GENERATION = False  # HACK FOR TESTING ONLY! GIVES GARBAGE RESULTS!
            if not HACK_SKIP_IMAGE_GENERATION:
                save_data = []
                exp_name = meta_infos[0]["experiment_name"] # MW hacked to add this
                assert len(prompt_texts) == len(generated_texts)

                ts = dt.strftime(dt.now(), format="%y%m%d%H%M%S")
                path_stub = f"rl4lm_exps/{exp_name}/image_generations/{split_name}/{ts}"
                
                for i in range(0, len(generated_texts), METRIC_BS): 
                    prompt_texts_chunk = prompt_texts[i:i+METRIC_BS]
                    generated_texts_chunk = generated_texts[i:i+METRIC_BS]
                    combined_texts_chunk = [ p + " "  + g.strip() for p, g in zip(
                        prompt_texts_chunk, generated_texts_chunk)]
                    meta_infos_chunk = meta_infos[i:i+METRIC_BS]
                    topics_chunk = [mi["topic"] for mi in meta_infos_chunk]
                    
                    actual_chunk_size = min(METRIC_BS, len(prompt_texts_chunk))
                    # always prompt better model with prompt_texts, worse can be either prompt + gen
                    # or the topic + gen
                    prompt_texts_worse_chunk = combined_texts_chunk
                    if self._use_topic_only_for_worse:
                        prompt_texts_worse_chunk = [tp + " " + g.strip() for tp, g in  
                                zip(topics_chunk, generated_texts_chunk)]
                    # generate METRIC_BS group of images
                    # goal is for the worst model + text generations to be similar to 
                    # the better model with the original prommpt
                    # Note: if you pass smaller width and height especially SD-v2.1 is terrible (not usable)
                    # better_generated_images = self._better_diffusion_pipeline(
                    #        prompt_texts_chunk).images

                    # BETTER_SPLIT_NAME = split_name
                    # BETTER_SPLIT_NAME = "train"
                    BETTER_SPLIT_NAME = "track_hack"
                    better_images_path = f'/home/ubuntu/RL4LMs/rl4lm_exps/image_generations/{BETTER_SPLIT_NAME}/better_images'
                    better_generated_images = []
                    for j in range(0, actual_chunk_size):
                        bi = Image.open(os.path.join(better_images_path, 
                            f"{meta_infos_chunk[j]['id']}_better.png"))
                        better_generated_images.append(bi)
                    worse_generated_images = self._worse_diffusion_pipeline(
                            prompt_texts_worse_chunk).images
                    
                    image_similarity_scores_chunk = self._clip_score_image_image(
                        better_generated_images, worse_generated_images)
                    image_similarity_scores_chunk = self.normalize_rewards(
                            image_similarity_scores_chunk)

                    image_similarity_scores[i:i+METRIC_BS] = image_similarity_scores_chunk

                    if split_name == "val" or split_name == "test":
                        # better_images_path = f"{path_stub}/better_images/"
                        worse_images_path = f"{path_stub}/worse_images/"
                        # if not os.path.exists(better_images_path):
                        #    print(f"Had to create path {better_images_path}")
                        #    os.makedirs(better_images_path)
                        if not os.path.exists(worse_images_path):
                            print(f"Had to create path: {worse_images_path}")
                            os.makedirs(worse_images_path)

                        for i_chunk in range(actual_chunk_size):
                            image_id = meta_infos_chunk[i_chunk]["id"]
                            better_path = os.path.join(better_images_path, f"{image_id}_better.png")
                            worse_path = os.path.join(worse_images_path, f"{image_id}_worse.png")
                            # better_generated_images[i_chunk].save(better_path)
                            worse_generated_images[i_chunk].save(worse_path)
                        
                            generation_data = {
                                "split": split_name,
                                "experiment_name": exp_name,
                                "image_id": image_id,
                                "better_image_path": better_path,
                                "worse_image_path": worse_path,
                                "text_prompt": prompt_texts_chunk[i_chunk],
                                "text_generation": generated_texts_chunk[i_chunk],
                                "text_combined": combined_texts_chunk[i_chunk],
                                "better_prompt": prompt_texts_chunk[i_chunk],
                                "worse_prompt" : prompt_texts_worse_chunk[i_chunk],
                                "topic": topics_chunk[i_chunk],
                                "image_similarity_score": image_similarity_scores_chunk[i_chunk].item()
                            }
                            generation_data_path = f"{path_stub}/generation_data.jsonl"
                            with open(generation_data_path, "a") as f:
                                f.write(json.dumps(generation_data) + '/n')

                        print(f"Saved {METRIC_BS} each of {split_name} images to better: {better_images_path} and worse: {worse_images_path} and generation data to: {generation_data_path}")
            else:
                print("WARNING: SKIPPING IMAGE GENERATION!! RESULTS WILL BE GARBAGE!")
                image_similarity_scores = torch.ones((len(generated_texts), 1))*0.01

            print(f"Normalized (stretched) metric calculated image_similarity_scores: {image_similarity_scores}, generated_texts: {generated_texts}") 
        image_similarity_scores = image_similarity_scores.cpu().numpy()
        batch_mean_score = float(np.mean(image_similarity_scores))
        metric_dict = {"diffusion_image_similarity_score": (image_similarity_scores.tolist(), batch_mean_score)}

        return metric_dict


class GPT2Perplexity(BaseMetric):
    def __init__(
        self,
        stride: int,
    ) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"

        self._gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self._gpt2_model.to(self._device)
        self._stride = stride

    def get_device(self, model: PreTrainedModel):
        try:
            return model.transformer.first_device
        except:
            return model.device

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:


        input_texts = [p + " " + g.strip() for (p, g) in zip(prompt_texts, generated_texts)]
        encodings = self._tokenizer(input_texts, return_tensors="pt")

        with torch.no_grad():
            # This could should run but the problem is that the probabilities are super small
            # Maybe use the beam search length penalty, but unclear how to map it from 0 to 10  
            outputs = self._gpt2_model(**encodings.to(self._device))
            nll_probs = torch.gather(outputs["logits"], dim=-1, index=encodings["input_ids"].unsqueeze(1))
            seq_lengths = float(encodings["input_ids"].shape[-1])
            avg_nll_prob =   torch.sum(nll_probs) / seq_lengths
            probs = torch.exp(avg_nll_prob)

        return {
            "fluency_metrics/gpt2_ppl": (
                probs,
                np.mean(probs)
            )
        }


if __name__ == "__main__":

    prompt_texts = ["1", "2"]
    gen_texts = [
        "The dog is the boy's cat.",
        "A boy is picking apples from trees and put them into bags.",
    ]
    reference_texts = [
        ["NOT USED"], ["NOT USED"]
    ]
    # reference_texts = [
    #     ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
    #     ["A boy is picking apples from trees."],
    # ]
    metric = DiffusionImageGenerationSimilarityMetric()
    print(metric.compute(prompt_texts, gen_texts, reference_texts))

    metric = SpiceMetric()
    print(metric.compute(prompt_texts, gen_texts, reference_texts))
