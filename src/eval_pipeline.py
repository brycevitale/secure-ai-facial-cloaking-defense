# Evaluation script will run after Fawkes finishes generating cloaked images.
# It will compute FaceNet embeddings and measure:
# - clean accuracy
# - edited accuracy (jpeg+blur)
# - cloaked accuracy
# - cloaked + jpeg defense
# - cloaked + blur defense

def main():
    print("TODO: run evaluation after cloaked images are ready.")
    print("Expected folders:")
    print("- data/faces_clean")
    print("- data/faces_edited")
    print("- data/faces_cloaked (or faces_cloaked_flat)")

if __name__ == "__main__":
    main()
