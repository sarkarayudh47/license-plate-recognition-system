from ultralytics import YOLO
model = YOLO("yolov8n.yaml")

train_results = model.train(
    data=r"C:\Users\AYUDH\Desktop\Computer_Vision\LicensePlate_Detection\licensePlateModel\data.yaml",  
    epochs=100,  
)
metrics = model.val()

# Perform object detection on an image
# results = model("path/to/image.jpg")  # Predict on an image
# results[0].show()  # Display results

# Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model