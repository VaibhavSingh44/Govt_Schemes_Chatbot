from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


if __name__ == "__main__":

    pdf_path = Path("Knowledge_base/test.pdf")

    pages = extract_text_from_pdf(pdf_path)

    print(f"Pages extracted: {len(pages)}")

    for page in pages[:2]:
        print("\n--- PAGE", page["page"], "---")
        print(page["text"][:1000])