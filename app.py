from pathlib import Path
import helper
import settings

def main():
    model_path = Path(settings.DETECTION_MODEL)
    print("Loading model...")
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        print(f"Unable to load model. Check the specified path: {model_path}")
        print(ex)
        return

    print("Starting Intelligent Waste Segregation System")
    print("Press 'q' to quit the webcam stream.")
    helper.play_webcam(model)

if __name__ == "__main__":
    main()
