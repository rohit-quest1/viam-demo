import asyncio
import datetime
from viam.rpc.dial import DialOptions
from viam.app.viam_client import ViamClient
from dotenv import load_dotenv
import os

def connect() -> ViamClient:
    """Establish connection with Viam platform."""
    dial_options = DialOptions.with_api_key(os.getenv('VIAM_API_KEY'), os.getenv('VIAM_API_KEY_ID'))
    return ViamClient.create_from_dial_options(dial_options)

async def upload_sensor_data(machine, sensor_data: dict):
    """Upload sensor data dictionary to Viam's platform."""
    data_client = machine.data_client  # Initialize data client
    time_requested = datetime.datetime.now()
    time_received = datetime.datetime.now()

    file_id = await data_client.tabular_data_capture_upload(
        part_id=os.getenv('VIAM_PART_ID'),
        component_type='rdk:component:sensor',
        component_name='sensor-1',
        method_name='Readings',
        tags=["sensor_data"],
        data_request_times=[(time_requested, time_received)],
        tabular_data=[{'readings': sensor_data}]
    )
    print(f"Sensor data uploaded successfully! File ID: {file_id}")

async def upload_data(sensor_data: dict):
    """Utility method to connect and upload sensor data."""
    load_dotenv()
    machine = await connect()
    await upload_sensor_data(machine, sensor_data)
    machine.close()
