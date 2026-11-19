from app.ingest import build_index, load_pdfs, split

if __name__ == "__main__":
    build_index(split(load_pdfs()))