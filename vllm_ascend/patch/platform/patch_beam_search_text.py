#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

from vllm.entrypoints.llm import LLM

_original_beam_search = LLM.beam_search


def _patched_beam_search(self, prompts, params, lora_request=None, use_tqdm=False, concurrency_limit=None):
    outputs = _original_beam_search(
        self,
        prompts,
        params,
        lora_request=lora_request,
        use_tqdm=use_tqdm,
        concurrency_limit=concurrency_limit,
    )

    tokenizer = self.renderer.get_tokenizer()

    for output in outputs:
        for beam in output.sequences:
            decoder_prompt = (
                beam.orig_prompt
                if beam.orig_prompt["type"] != "enc_dec"
                else beam.orig_prompt["decoder_prompt"]
            )
            prompt_text = decoder_prompt.get("prompt") or ""
            prompt_len = len(decoder_prompt["prompt_token_ids"])
            generated_text = tokenizer.decode(beam.tokens[prompt_len:])
            beam.text = prompt_text + generated_text

    return outputs


LLM.beam_search = _patched_beam_search
