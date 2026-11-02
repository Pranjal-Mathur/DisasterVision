from ultralytics import YOLO

def main():

    model = YOLO(
        r"D:\Projects\DeepLearning\YOLO_Disaster_Project\runs\xbd_segmentation_fast4\weights\best.pt"
    )

    metrics = model.val(

        data=r"D:\Projects\DeepLearning\YOLO_Disaster_Project\seg_data.yaml",

        split="val",

        save_json=True,

        workers=0,

        plots=True
    )

    print(metrics)


if __name__ == "__main__":

    main()