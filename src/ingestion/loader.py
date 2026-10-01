import pymupdf
import json
import os
from typing import List
from pydantic import BaseModel

class SlideData(BaseModel):
    
    """
    Schema that holds raw data extracted from a single lecture slide
    
    """
    slide_num: int
    text: str
    image_path: str
    pdf_name: str


def load_lecture_slides(pdf_path: str,
                        output_dir: str,
                        dpi: int) -> List[SlideData]:

    doc = pymupdf.open(pdf_path)
    pdf_name = os.path.basename(pdf_path)

    slides = []

    for page in range(len(doc)):
        
        slide_num = page + 1

        pix = doc[page].get_pixmap(dpi=dpi)

        text = doc[page].get_text("text").strip()

        image_filename = (f"{os.path.splitext(pdf_name)[0]}_slide_{slide_num}.png")
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)

        slides.append(SlideData(
            slide_num=slide_num,
            text=text,
            image_path=image_path,
            pdf_name=pdf_name
        ))

    doc.close()

    return slides

def main():
    example_pdf = "/Users/wayne/Personal_Project/rag/data/raw/Lectures/01_Introduction.pdf"
    output_dir = "/Users/wayne/Personal_Project/rag/data/processed/Lectures"

    slides = load_lecture_slides(example_pdf, output_dir, dpi=200)

    with open(f"{output_dir}/slides_info.json", "w") as f:
        json.dump([slide.model_dump() for slide in slides], f, indent=2)
        # Able to do model_dump() because of pydantic class BaseModel
        # we can then reconstruct it back to the object by doing 
            # slides = SlideData.model_validate_json(saved_json)

if __name__=="__main__":
    main()
