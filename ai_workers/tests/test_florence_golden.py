from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor

from ai_workers.modules.visual_v2.florence_runtime import (
    FlorenceDeterminism,
    resolve_florence_runtime,
    verify_florence_model,
)


@unittest.skipUnless(
    os.environ.get("RUN_FLORENCE_SMOKE_TEST") == "1",
    "Set RUN_FLORENCE_SMOKE_TEST=1 to run the real Florence checkpoint.",
)
class FlorenceGoldenCaptionTests(unittest.TestCase):
    def test_red_square_caption_matches_across_supported_cpu_hosts(self):
        model_dir = Path(__file__).resolve().parents[1] / "modules" / "visual_v2" / "florence2_vendor"
        runtime = resolve_florence_runtime("cpu")
        verify_florence_model(model_dir)
        guard = FlorenceDeterminism(runtime)

        processor = None
        model = None
        guard.enable()
        try:
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                dtype=runtime.dtype,
                trust_remote_code=True,
                attn_implementation=runtime.attention_implementation,
            ).to(runtime.device)
            model.eval()

            image = Image.new("RGB", (768, 768), "white")
            ImageDraw.Draw(image).rectangle((184, 184, 584, 584), fill=(220, 30, 30))
            inputs = processor(text="<CAPTION>", images=image, return_tensors="pt")
            inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=runtime.dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=64,
                    num_beams=3,
                    do_sample=False,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            parsed = processor.post_process_generation(
                generated_text,
                task="<CAPTION>",
                image_size=image.size,
            )
            self.assertEqual(
                parsed["<CAPTION>"].strip(),
                "a red square on a white background",
            )
        finally:
            del model
            del processor
            guard.restore()


if __name__ == "__main__":
    unittest.main()
