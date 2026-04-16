import pymupdf4llm

md_text = pymupdf4llm.to_markdown("/workspaces/Rag_Chat_Assignement/Documens/pdfs/" \
                                "SC_2025_36 Introduction of LPT 1612g with 3.8 SGI TC CNG BS6 Ph2.pdf",
                                write_images=True,
                                page_chunks = True,
                                image_path="images_folder", # Directory where images will be saved
                                image_format="png",         # Supported: "png", "jpg", etc.
                                force_text=False,           # stop text extraction if images are present
                                dpi=150  )                  # Set resolution (default is 150)

# Now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
print("Markdown saved to output.md")