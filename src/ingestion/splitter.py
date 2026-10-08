import ollama
from pydantic import BaseModel
from typing import List
import time
import textwrap
import glob
import os
import json
from config import VISION_MODEL
from ingestion.loader import SlideData
from config import PROCESSED_DIR

class SlideChunk(BaseModel):
    
    """
    A processed slide ready for embedding
    
    """
    slide_num: int
    combined_text: str
    visual_description: str
    image_path: str
    metadata: dict

def generate_slide_description(
        image_path: str,
        raw_text: str
) -> str:
    # base64_image = encode_image(image_path)

    prompt = textwrap.dedent(f"""
    
    You are describing visual content on a lecture slide for a search index.

    Text already extracted from this slide (reference only):
    <extracted_text>
    {raw_text}
    </extracted_text>

    Look at the slide image. Decide whether it contains a diagram, chart, graph, formula, table, or informative image.
    Logos, backgrounds, decorative pictures, and slide numbers do not count.

    If there is none, answer with the single word: NONE

    If there is one, write one short paragraph explaining what the visual shows and what it means.
    Mention axis labels, components and how they connect, or formula variables where relevant.
    Describe only what you can clearly see. Use the extracted text for context but do not repeat it.

    Example answer for a slide with only bullet points and a university logo:
    NONE

    Example answer for a slide with a chart:
    A line graph of training and validation loss against epochs. Training loss falls steadily, while validation loss falls until about epoch 10 and then rises, illustrating overfitting.

    Your answer:

    """).strip()

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
        options={"temperature": 0.0},
    )
    return response['message']['content']

def process_slides_with_context(
        raw_slides: List,
) -> List[SlideChunk]:
    
    processed_chunk = []

    total = len(raw_slides)

    print(f"Starting Processing for {total} Slides")

    for idx, slide in enumerate(raw_slides):
        
        start_time = time.time()

        vision_desc = generate_slide_description(
            slide.image_path,
            slide.text,
        )

        elapsed = time.time() - start_time

        prev_title = (f"Slide {raw_slides[idx-1].slide_num}" if idx > 0 else "Beginning of deck")

        next_title = (f"Slide {raw_slides[idx+1].slide_num}" if idx < total - 1 else "End of deck")

        if vision_desc.strip().upper() == "NONE":

            combined_text = f"""
                [CONTEXT]: Previous slide was {prev_title}. Next slide is {next_title}.
                [SLIDE {slide.slide_num} TEXT]:
                {slide.text}
                """
        else:
            combined_text = f"""
                [CONTEXT]: Previous slide was {prev_title}. Next slide is {next_title}.
                [SLIDE {slide.slide_num} TEXT]:
                {slide.text}
                [VISUAL DESCRIPTION]:
                {vision_desc}
                """

        processed_chunk.append(
            SlideChunk(
                slide_num=slide.slide_num,
                combined_text=combined_text,
                visual_description=vision_desc,
                image_path=slide.image_path,
                metadata={
                    "pdf_name": slide.pdf_name,
                    "slide_num": slide.slide_num,
                    "image_path": slide.image_path
                },
            )
        )

        print(elapsed)

    return processed_chunk
    

def main():

    all_slides_json = glob.glob(os.path.join(PROCESSED_DIR, "*_slides.json"))

    for slides_path in all_slides_json:
        
        out_path = slides_path.replace("_slides.json", "_chunks.json")

        if os.path.exists(out_path):
            print(f"Skipping {os.path.basename(slides_path)} (already processed)")
            continue

        with open(slides_path) as f:
            raw_slides = [SlideData.model_validate(item) for item in json.load(f)]

        chunks = process_slides_with_context(raw_slides)

        with open(out_path, "w") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=2)

        print(f"Saved {len(chunks)} chunks to {out_path}")

if __name__ == "__main__":
    main()