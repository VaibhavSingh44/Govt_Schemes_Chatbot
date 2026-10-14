def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    sample_text = """
    Cooperative societies are voluntary organizations formed by individuals
    who come together to achieve common economic and social objectives.
    Cooperative governance involves democratic participation, member rights,
    responsibilities, elections, management committees and financial
    accountability. Members may participate in general meetings and vote
    according to the applicable cooperative law and by-laws.
    """ * 20

    chunks = chunk_text(sample_text)

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3], start=1):

        print(f"\n--- CHUNK {i} ---")
        print(chunk[:1000])