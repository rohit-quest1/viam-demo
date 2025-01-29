import asyncio
from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.camera import Camera
from viam.services.vision import VisionClient
from viam.services.mlmodel import MLModelClient
import os
from io import BytesIO
import pytesseract
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import numpy as np
from PIL import Image, ImageEnhance
from viam.media.video import CameraMimeType
import cv2
from dotenv import load_dotenv
load_dotenv()

AZURE_SUB_KEY=os.getenv("AZURE_SUB_KEY")
AZURE_ENDPOINT=os.getenv("AZURE_ENDPOINT")
VIAM_API_KEY=os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID=os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ID=os.getenv("VIAM_MACHINE_ID")
SLEEP_TIMER=os.getenv("SLEEP_TIMER")

async def ocr():
    subscription_key = AZURE_SUB_KEY
    endpoint = AZURE_ENDPOINT

    computervision_client = ComputerVisionClient(
        endpoint, CognitiveServicesCredentials(subscription_key))

    try:
        # Verify the image exists
        if not os.path.exists('cropped_image.png'):
            print("Error: cropped_image.png not found")
            return

        # Print image size for debugging
        image_size = os.path.getsize('cropped_image.png')
        print(f"Image size: {image_size} bytes")

        # Read the image file and create BytesIO object
        with open('cropped_image.png', 'rb') as image_file:
            image_data = BytesIO(image_file.read())
            
        # Analyze the image first to confirm it's readable
        image_data.seek(0)  # Reset buffer position
        vision_analysis = computervision_client.analyze_image_in_stream(
            image_data,
            visual_features=['Description']
        )
        print("Image analysis succeeded, proceeding with OCR")
            
        # Reset buffer position and perform OCR
        image_data.seek(0)
        read_response = computervision_client.read_in_stream(
            image_data,
            language="en",
            raw=True
        )
            
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        print(f"OCR operation initiated with ID: {operation_id}")

        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            print(f"OCR Status: {read_result.status}")
            await asyncio.sleep(1)

        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    print(f"Detected line: {line.text}")
        else:
            print(f"OCR operation failed with status: {read_result.status}")
            
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        raise



async def color_det():
    global final_col
    car_img = cv2.imread('original_image.png')

    hsv_car = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'White': [(0, 0, 231), (180, 18, 255)],
        'Red': [(0, 0, 0), (180, 255, 30)],
        'Red': [(0, 70, 50), (10, 255, 255)],
        'Red': [(11, 70, 50), (20, 255, 255)],
        'Yellow': [(21, 70, 50), (30, 255, 255)],
        'Green': [(31, 70, 50), (90, 255, 255)],
        'Blue': [(100, 50, 50), (130, 255, 255)],
        'Purple': [(131, 70, 50), (170, 255, 255)],
        'Red': [(171, 70, 50), (179, 255, 255)],
    }

    a = {}
    for color_name, color_range in color_ranges.items():
        mask = cv2.inRange(hsv_car, color_range[0], color_range[1])
        pixel_count = cv2.countNonZero(mask)
        print(pixel_count)
        a[color_name] = pixel_count
    final_col = max(a, key=a.get)
    print("Color of Original Image: ",final_col)
    return (final_col)



async def connect():
    opts = RobotClient.Options.with_api_key( 
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID
    )
    return await RobotClient.at_address(VIAM_MACHINE_ID, opts)


async def main():
    machine = await connect()

    print('Resources:')
    print(machine.resource_names)
    
    camera_1 = Camera.from_robot(machine, "camera-1")
    vision_2 = VisionClient.from_robot(machine, "vision-1")

    try:
        while True:
            camera_1_return_value = await camera_1.get_image()
            print(f"camera-1 get_image return value: {camera_1_return_value}")

            detections = await vision_2.get_detections(camera_1_return_value)

            if detections:
                for detection in detections:
                    if detection.confidence > 0.75:
                        print(f"Detection confidence: {detection.confidence}")
                        print(f"x_min: {detection.x_min}")
                        print(f"y_min: {detection.y_min}")
                        print(f"x_max: {detection.x_max}")
                        print(f"y_max: {detection.y_max}")
                        print(f"confidence: {detection.confidence}")
                        print(f"class_name: {detection.class_name!r}")
                        
                        standard_frame = camera_1_return_value.data
                        image = Image.open(BytesIO(standard_frame))
                        image.save('original_image.png', 'PNG', quality=100)
                        
                        # Get original image dimensions
                        width, height = image.size
                        
                        # Calculate the current detection box size
                        box_width = detection.x_max - detection.x_min
                        box_height = detection.y_max - detection.y_min
                        
                        # Calculate padding based on box size
                        padding_x = max(int(box_width * 0.3), 20)  # At least 20 pixels or 30% of width
                        padding_y = max(int(box_height * 0.3), 20)  # At least 20 pixels or 30% of height
                        
                        # Calculate new coordinates with padding
                        x_min = max(0, int(detection.x_min - padding_x))
                        y_min = max(0, int(detection.y_min - padding_y))
                        x_max = min(width, int(detection.x_max + padding_x))
                        y_max = min(height, int(detection.y_max + padding_y))
                        
                        # Crop the image
                        cropped_image = image.crop((x_min, y_min, x_max, y_max))
                        
                        # Get dimensions of cropped image
                        crop_width, crop_height = cropped_image.size
                        print(f"Cropped image size: {crop_width}x{crop_height}")
                        
                        # Resize if either dimension is less than 50 pixels
                        if crop_width < 50 or crop_height < 50:
                            # Calculate scaling factor to make smallest dimension 50 pixels
                            scale = max(50 / crop_width, 50 / crop_height)
                            new_width = int(crop_width * scale)
                            new_height = int(crop_height * scale)
                            cropped_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            print(f"Resized image to: {new_width}x{new_height}")
                        
                        # Enhance the image for better OCR
                        enhancer = ImageEnhance.Contrast(cropped_image)
                        cropped_image = enhancer.enhance(1.5)
                        
                        enhancer = ImageEnhance.Sharpness(cropped_image)
                        cropped_image = enhancer.enhance(1.5)
                        
                        # Convert to RGB if not already
                        if cropped_image.mode != 'RGB':
                            cropped_image = cropped_image.convert('RGB')
                        
                        # Save with high quality
                        cropped_image.save('cropped_image.png', 'PNG', quality=100)
                        
                        print("Image saved, attempting OCR...")

                        # Uncomment when needed
                        await ocr()
                        await color_det()

            await asyncio.sleep(int(SLEEP_TIMER))

    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        await machine.close()


if __name__ == '__main__':
    asyncio.run(main())
