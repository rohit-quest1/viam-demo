import asyncio

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.camera import Camera
from viam.services.vision import VisionClient
from viam.services.mlmodel import MLModelClient

from PIL import Image
from io import BytesIO
import pytesseract
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
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
    vision_2 = VisionClient.from_robot(machine, "vision-2")

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
                            preprocessing('cropped_image.png')
                            
                            text = pytesseract.image_to_string('preprocessed_image.png')
                            print(f"OCR Text: {text}")

            await asyncio.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        await machine.close()
        
def preprocessing(impath):
    img = cv2.imread(impath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
    edged = cv2.Canny(bfilter, 30, 200) 
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints) 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8) 
    new_image = cv2.drawContours(mask, [location], 0,255, -1) 
    new_image = cv2.bitwise_and(img, img, mask=mask) 
    
    (x,y) = np.where(mask==255) 
    (x1, y1) = (np.min(x), np.min(y)) 
    (x2, y2) = (np.max(x), np.max(y)) 
    cropped_image = gray[x1:x2+1, y1:y2+1] 
    plt.imsave('preprocessed_image.png',cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    

if __name__ == '__main__':
    asyncio.run(main())