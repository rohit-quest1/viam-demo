import asyncio

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.camera import Camera
from viam.services.vision import VisionClient
from viam.services.mlmodel import MLModelClient

from PIL import Image
from io import BytesIO
import pytesseract

import numpy as np
from viam.media.video import CameraMimeType

async def connect():
    opts = RobotClient.Options.with_api_key( 
        api_key='69bbi2ti94asqlsrjfklutgfdlyubi41',
        api_key_id='fe348cc0-4aa0-4654-84d4-826cf0632e4b'
    )
    return await RobotClient.at_address('mylaptop-main.5v3r0ehp5p.viam.cloud', opts)

async def main():
    machine = await connect()

    print('Resources:')
    print(machine.resource_names)
    
    # camera-1
    camera_1 = Camera.from_robot(machine, "camera-1")
    mlmodel_1 = MLModelClient.from_robot(machine, "mlmodel-1")
    vision_2 = VisionClient.from_robot(machine, "vision-1")

    try:
        while True:
            camera_1_return_value = await camera_1.get_image()
            print(f"camera-1 get_image return value: {camera_1_return_value}")

            # mlmodel-1
            # mlmodel_1_return_value = await mlmodel_1.metadata()
            # print(f"mlmodel-1 metadata return value: {mlmodel_1_return_value}")

            # vision-2
            # vision_2_return_value = await vision_2.get_properties()
            detections = await vision_2.get_detections(camera_1_return_value)

            # if(detections):
            #     for detection in detections:
            #         if(detection.confidence > 0.75):

            #             print(f"vision-2 get_properties return value: {detection.confidence} \n")
            #             print(detection)
            #             standard_frame = camera_1_return_value.data
            #             image = Image.open(BytesIO(standard_frame))
                        
            #             # Save the image locally
            #             image.save('depth_image.png')


            if detections:
                    for detection in detections:
                        if detection.confidence > 0.75:
                            print(f"Detection confidence: {detection.confidence} \n")
                            print(detection)
                            
                            # Get the standard frame (depth image data)
                            standard_frame = camera_1_return_value.data
                            image = Image.open(BytesIO(standard_frame))
                            
                            # Extract bounding box coordinates
                            x_min = detection.x_min
                            y_min = detection.y_min
                            x_max = detection.x_max
                            y_max = detection.y_max
                            
                            # Crop the image using the bounding box
                            cropped_image = image.crop((x_min, y_min, x_max, y_max))
                            
                            # Save the cropped image locally
                            cropped_image.save('cropped_image.png')
                            text = pytesseract.image_to_string(cropped_image)
                            print(f"OCR Text: {text}")

            await asyncio.sleep(5)

      
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        await machine.close()

if __name__ == '__main__':
    asyncio.run(main())
