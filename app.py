import asyncio

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.camera import Camera
from viam.services.vision import VisionClient

async def connect():
    opts = RobotClient.Options.with_api_key( 
        api_key='VIAM_API_KEY',
        api_key_id='VIAM_API_KEY_ID'
    )
    return await RobotClient.at_address('YOUR_MACHINE_ADDRESS', opts)

async def main():
    machine = await connect()

    print('Resources:')
    print(machine.resource_names)
    
    camera_1 = Camera.from_robot(machine, "camera-1")
    vision_1 = VisionClient.from_robot(machine, "vision-1")
    
    try:
        while True:
            camera_1_return_value = await camera_1.get_image()
            detections = await vision_1.get_detections(camera_1_return_value)
            
            if(detections):
                for detection in detections:
                    if detection.class_name == "red":
                        print(f"Danger detected! Located at coordinates: "
                            f"({detection.x_min}, {detection.y_min}) to "
                            f"({detection.x_max}, {detection.y_max})")
            else:
                print("No Color Detected")
            
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopping color detection...")
    finally:
        await machine.close()

if __name__ == '__main__':
    asyncio.run(main())