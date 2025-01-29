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
import boto3
import os
import random
import datetime
import uuid
import json
import aiofiles
from botocore.exceptions import NoCredentialsError
from serpapi import GoogleSearch
from viam_uploader import upload_data
from mail import send_alert_email

from dotenv import load_dotenv
load_dotenv()


AWS_BUCKET_NAME=os.getenv('S3_AWS_BUCKET_NAME')
AWS_ACCESS_KEY=os.getenv('S3_AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY=  os.getenv('S3_AWS_SECRET_ACCESS_KEY')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
AZURE_SUB_KEY = os.getenv("AZURE_SUB_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
except Exception as e:
    print(f"Error initializing S3 client: {e}")
    raise


with open('data.json','r') as f:
    data = json.load(f)
       
with open('corpus.json','r') as f:
    corpus_data = json.load(f)
    
async def ocr(fpath):
    computervision_client = ComputerVisionClient(
        AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_SUB_KEY))

    try:
        # Verify the image exists
        if not os.path.exists(fpath):
            print(f"Error:{fpath} not found")
            return

        # Print image size for debugging
        image_size = os.path.getsize(fpath)
        print(f"Image size: {image_size} bytes")

        # Read the image file and create BytesIO object
        with open(fpath, 'rb') as image_file:
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
            print(f"TEXT : {[text_result.lines for text_result in read_result.analyze_result.read_results]}")
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    return line.text.strip()
                
        else:
            print(f"OCR operation failed with status: {read_result.status}")
            
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        raise

async def color_det(fpath):
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
    return final_col

async def generate_unique_geo_filename(prefix='image', extension='png'):
    """
        Filename format: prefix_TIMESTAMP_LAT_LON_UUID.extension
    """
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)
    
    lat_dir = 'N' if latitude >= 0 else 'S'
    lon_dir = 'E' if longitude >= 0 else 'W'
    lat_str = f"{abs(latitude):.6f}".replace('.', '_') + lat_dir
    lon_str = f"{abs(longitude):.6f}".replace('.', '_') + lon_dir
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    unique_id = uuid.uuid4().hex[:8]
    
    return f"images/{prefix}_{timestamp}_{lat_str}_{lon_str}_{unique_id}.{extension.lstrip('.')}"

async def get_s3_url(fpath):
    """Uploads file to S3 and returns public URL"""
    image_url = None
    try:
        # Upload the file
        s3.upload_file(
            fpath,
            AWS_BUCKET_NAME,
            fpath,
            ExtraArgs={
                'ContentType': 'image/png',
                'ACL': 'public-read',
                'ContentDisposition': 'inline' 
            }
        )
        
        # Construct URL using bucket's region
        location = s3.get_bucket_location(Bucket=AWS_BUCKET_NAME)['LocationConstraint']
        region = location if location else 'us-east-1'
        image_url = f"https://{AWS_BUCKET_NAME}.s3.{region}.amazonaws.com/{fpath}"
        
    except FileNotFoundError:
        print(f"Error: File {fpath} not found")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
    
    return image_url

async def extract_model(fpath):
    global car_type
    
    car_type = None
    image_url = await get_s3_url(fpath)
    params = {
        "engine": "google_reverse_image",
        "image_url": image_url,  
        "api_key": SERPAPI_KEY 
    }
    print("API Parameters:", params)

    # try:
    search = GoogleSearch(params)
    results = search.get_dict()
    print(results)
    await asyncio.sleep(3)
    
    car_type = results["knowledge_graph"]["title"]
    print("The model of the car from the given image is ",car_type)
    return car_type,image_url
    # except Exception as e:
    #     print(f"SerpAPI Error: {e}")

async def write_json_async(data):
    async with aiofiles.open('data.json', 'w') as f:
        await f.write(json.dumps(data))
        
async def connect():
    
    opts = RobotClient.Options.with_api_key( 
        api_key= os.getenv('VIAM_API_KEY'),
        api_key_id= os.getenv('VIAM_API_KEY_ID')
    )
    return await RobotClient.at_address('demo-mac-main.fnhngyx1au.viam.cloud', opts)

async def compare_vehicle_data(predicted):
    # flag = False  # Flag to indicate mismatch
    # discrepancies = {}
    
    # if corpus_data["vehicle_number"] != predicted["vehicle_number"]:
    #     discrepancies["vehicle_number"] = {
    #         "actual": corpus_data["vehicle_number"],
    #         "predicted": predicted["vehicle_number"]
    #     }
    #     flag = True
        
    # # Compare actual vs expected parameters
    # actual_params = corpus_data.get("actual_params", {})
    # expected_params = predicted.get("expected_params", {})
    
    # for key in ["model", "colour"]:
    #     actual_value = actual_params.get(key)
    #     predicted_value = expected_params.get("make" if key == "model" else key)
        
    #     if actual_value != predicted_value:
    #         discrepancies[key] = {
    #             "actual": actual_value,
    #             "predicted": predicted_value
    #         }
    #         flag = True 
    print("CORPUS :",corpus_data.keys(),"v_no :",predicted['vehicle_number'].strip())
    if predicted['vehicle_number'].strip() in corpus_data.keys():
        return False
    else:
        return True

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
                        fname = await generate_unique_geo_filename(prefix="og_img")
                        image.save(fname, 'PNG', quality=100)
                        
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
                        
                        rv_cropped_image = image.crop(
                            (max(0, int(detection.x_min - max(int(box_width * 0.5), 50))),
                                max(0, int(detection.y_min - max(int(box_height * 0.5), 50))),
                                min(width, int(detection.x_max + max(int(box_width * 0.5), 50))),
                                min(height, int(detection.y_max + max(int(box_height * 0.5), 50)))
                                )
                        )
                        
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
                        
                        enhancer = ImageEnhance.Contrast(rv_cropped_image)
                        rv_cropped_image = enhancer.enhance(1.5)
                        
                        enhancer = ImageEnhance.Sharpness(cropped_image)
                        cropped_image = enhancer.enhance(1.5)
                        
                        enhancer = ImageEnhance.Sharpness(rv_cropped_image)
                        rv_cropped_image = enhancer.enhance(1.5)
                        
                        # Convert to RGB if not already
                        if cropped_image.mode != 'RGB':
                            cropped_image = cropped_image.convert('RGB')
                        
                        if rv_cropped_image.mode!= 'RGB':
                            rv_cropped_image = rv_cropped_image.convert('RGB') 
                            
                        # Save with high quality
                        cropped_image.save(fname.replace("og_img","crp_img"), 'PNG', quality=100)
                        # rv_cropped_image.save(fname.replace("og_img","only_car"),'PNG',quality=100)
                        print("Image saved, attempting OCR...")

                        # Uncomment when needed
                        license_plate_text = await ocr(fname.replace("og_img","crp_img"))
                        # car_color = await color_det(fname)
                        # car_model,image_url = await extract_model(fname)
                        car_color = "green"
                        car_model = "SUV"
                        
                        location = s3.get_bucket_location(Bucket=AWS_BUCKET_NAME)['LocationConstraint']
                        region = location if location else 'us-east-1'
                        image_url = f"https://{AWS_BUCKET_NAME}.s3.{region}.amazonaws.com/{fname}"
                        
                        license_plate_text = ''.join([char for char in license_plate_text if char.isalnum()])
                        license_plate_text = license_plate_text.strip()
                        license_plate_text = license_plate_text.upper()
                        
                        pred_data = {
                            'vehicle_number':license_plate_text,
                            'expected_params': {
                                'make': car_model,
                                'colour':car_color
                            },
                            'timestamp':datetime.datetime.strptime('_'.join(fname.split('_')[2:5]),"%Y%m%d_%H%M%S_%f")
                        }
                        
                        mismatch_flag = await compare_vehicle_data(pred_data)
                        new_data = data.copy()
                        
                        record = {
                                'mismatch':mismatch_flag,
                                'vehicle_number':license_plate_text,
                                'expected_params': {
                                    'make': car_model,
                                    'colour':car_color
                                },
                                'image_url':image_url,
                                'actual_params':corpus_data.get(license_plate_text),
                                'timestamp':str(datetime.datetime.strptime('_'.join(fname.split('_')[2:5]),"%Y%m%d_%H%M%S_%f"))
                        }
                        if mismatch_flag:
                            await upload_data(record)
                            
                            new_data.append(record)

                            await write_json_async(new_data)

                            await send_alert_email(record)
                            
                            
                        
            await asyncio.sleep(5)
            
    except Exception as e:
        print(e)
        
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        await machine.close()


if __name__ == '__main__':
    asyncio.run(main())
