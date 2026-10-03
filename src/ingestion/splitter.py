import ollama
import base64
from pydantic import BaseModel
from typing import List
import time

# need to do ollama run qwen2.5-vl to add to your ollama list

class SlideChunk(BaseModel):
    
    """
    Schema that holds raw data extracted from a single lecture slide
    
    """
    slide_num: int
    combined_text: str
    visual_description: str
    image_path: str
    metadata: dict

def encode_image(image_path: str) -> str:
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def generate_slide_description(
        image_path: str,
        raw_text: str
) -> str:
    base64_image = encode_image(image_path)

    prompt = f"""
    
    You are analyzing a slide image from a lecture deck. 

    Text already extracted from slide:
    "{raw_text}"

    TASK:
    1. Identify if this slide contains non-text visual elements (e.g., diagrams, graphs, charts, flowcharts, formulas, formulas in images, logos, or photos).
    2. IF VISUAL ELEMENTS ARE PRESENT, inspect them using these steps before describing:
        - Panel Count: Count total distinct chart panels or images (e.g. 1 single plot, 2x2 grid).
        - Chart Identification: For each plot, explicitly verify if it uses individual dots (scatter plot), vertical blocks (histogram/bar chart), or lines (line chart).
        - Axes: Identify exact X-axis and Y-axis variable names.
        - Trend/Meaning: Summarize key trends or relationship shown across the visual elements.
    
    CRITICAL RULES:
        - Combine these observations into a single concise paragraph.
        - Do NOT re-summarize or transcribe the plain text already extracted above.
        - Do NOT mention slide numbers, page counters (e.g. "X/49"), or state what is missing.

    """

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image],
            }
        ],
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

        if vision_desc.strip().upper() == "NONE" or "NONE" in vision_desc.upper():
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

        print(f"Done {elapsed:.2f}s")

        print(combined_text)

    return processed_chunk
    

def main():

    from ingestion.loader import SlideData
    import json

    raw_text_path = "/Users/wayne/Personal_Project/rag/data/processed/Lectures/slides_info.json"

    with open(raw_text_path, "r") as f:
        data = json.load(f)

    raw_slides = [SlideData.model_validate(item) for item in data]

    print(f"Loading raw slide data from: {raw_text_path}")
    processed_slides = process_slides_with_context(raw_slides)

    print(f"Processed {len(processed_slides)} slides successfully.")

    output_path = "/Users/wayne/Personal_Project/rag/data/processed/Lectures/processed_chunks.json"
    
    chunks_dict = [chunk.model_dump() for chunk in processed_slides]
    
    with open(output_path, "w") as f:
        json.dump(chunks_dict, f, indent=2)

    print(f"Saved {len(processed_slides)} processed chunks to {output_path}")

if __name__ == "__main__":
    main()