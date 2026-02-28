from google.cloud import aiplatform

def main():
    # Just checking version
    print("Vertex AI SDK version:", aiplatform.__version__)

if __name__ == "__main__":
    main()
