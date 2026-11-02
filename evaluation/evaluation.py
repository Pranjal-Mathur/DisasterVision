from ultralytics import YOLO

def main():

    # Load trained model
    model = YOLO(
        r"D:\Projects\DeepLearning\YOLO_Disaster_Project\evaluation\best_xbd_yolo.pt"
    )

    # Run validation
    metrics = model.val(

        data="balanced_data.yaml",

        split="val",

        save_json=True,

        workers=0
    )

    print(metrics)


if __name__ == "__main__":

    main()