from pathlib import Path

from pdf_reader import extract_text_from_pdf
from chunker import chunk_text


current_dir = Path(__file__).resolve().parent
pdf_path = current_dir.parent/ "Knowledge_base" / "test.pdf"


def process_document(pdf_path):

    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []

    for page in pages:

        chunks = chunk_text(
            page["text"],
            chunk_size=500,
            overlap=100
        )

        for chunk_number, chunk in enumerate(chunks, start=1):

            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page["page"],
                        "chunk": chunk_number
                    }
                }
            )

    return all_chunks


if __name__ == "__main__":

    chunks = process_document(pdf_path)

    print("Total chunks:", len(chunks))

    for i, item in enumerate(chunks[:5], start=1):

        print(f"\n========== CHUNK {i} ==========")

        print("Source:", item["metadata"]["source"])
        print("Page:", item["metadata"]["page"])
        print("Chunk:", item["metadata"]["chunk"])

        print("\nText:")
        print(item["text"][:1000])