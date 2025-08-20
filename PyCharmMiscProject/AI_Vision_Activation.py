import asyncio
import time
import cv2
import numpy as np
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# -- CONFIGURATION --
VIAM_API_KEY_ID = 'cfdc3d38-3ad9-496f-87f9-8afde54a8865'
VIAM_API_KEY = 'qhvxlnk7kdqgk9t07ki16y8ehh3gw22g'
VIAM_MACHINE_URL = 'rpi-ai-main.q3cs6ir22n.viam.cloud'
CAMERA_NAME = 'EMEETSmartCamC9604KEMEETSusbxhcihcd01'

# -- CONNECT TO VIAM ROBOT --
async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID
    )
    return await RobotClient.at_address(VIAM_MACHINE_URL, opts)


# -- MAIN EXECUTION --
async def main():
    print("⏳ Connecting to robot...")
    robot = await connect()
    print("✅ Connected to robot!\n")

    # Load YOLOv8 model (pretrained)
    model = YOLO("yolov8n.pt")  # you can change to 'yolov8s.pt', 'yolov8m.pt', etc.

    camera = Camera.from_robot(robot, CAMERA_NAME)

    print("📡 Starting real-time object detection. Press 'q' to quit.")

    while True:
        # Get image from Viam camera
        viam_image = await camera.get_image()
        img = Image.open(BytesIO(viam_image.data)).convert("RGB")
        frame = np.array(img)

        # Run YOLOv8 inference
        results = model(frame)[0]  # first result from batch

        # Draw detections
        annotated_frame = results.plot()

        # Show frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    await robot.close()
    cv2.destroyAllWindows()
    print("👋 Disconnected cleanly.")


# -- ENTRY POINT --
if __name__ == '__main__':
    asyncio.run(main())