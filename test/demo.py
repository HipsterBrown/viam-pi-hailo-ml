import asyncio
import os

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.services.vision import VisionClient

from viam.media.utils.pil import viam_to_pil_image

import cv2
import numpy as np


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.getenv("VIAM_API_KEY"),
        api_key_id=os.getenv("VIAM_API_KEY_ID"),
    )
    return await RobotClient.at_address(os.getenv("VIAM_MACHINE_ADDRESS"), opts)


async def main():
    machine = await connect()

    print("Resources:")
    print(machine.resource_names)

    # cam
    cam = Camera.from_robot(machine, "cam")
    cam_return_value = await cam.get_image()
    print(f"cam get_image return value: {cam_return_value}")

    # vision-1
    vision_1 = VisionClient.from_robot(machine, "vision-1")
    vision_1_return_value = await vision_1.get_properties()
    print(f"vision-1 get_properties return value: {vision_1_return_value}")

    vision_capture = await vision_1.get_detections(cam_return_value)
    print(f"Detected something: {vision_capture}")

    cv_image = cv2.cvtColor(
        np.array(viam_to_pil_image(cam_return_value)), cv2.COLOR_RGB2BGR
    )
    for detection in vision_capture:
        label = f"{detection.class_name} %{int(detection.confidence * 100)}"
        cv2.rectangle(
            cv_image,
            (detection.x_min, max(0, detection.y_min)),
            (detection.x_max, min(cam_return_value.height, detection.y_max)),
            (0, 255, 0, 0),
            2,
        )
        cv2.putText(
            cv_image,
            label,
            (detection.x_min + 5, detection.y_min + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite("detected.jpg", cv_image)

    # Don't forget to close the machine when you're done!
    await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
