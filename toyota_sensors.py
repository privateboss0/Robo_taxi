import carla
import random
import numpy as np
import time

HOST = 'localhost'
PORT = 2000
TIMEOUT = 10.0 

def process_semantic_camera(image):

    image.convert(carla.ColorConverter.CityScapesPalette) 
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4)) 
    semantic_tags = array[:, :, 2] 

    
    print(f"Semantic Cam Frame: {image.frame}, Timestamp: {image.timestamp}, Unique Tags: {np.unique(semantic_tags)}")

def process_semantic_lidar(lidar_data):

    print(f"Semantic Lidar Frame: {lidar_data.frame}, Timestamp: {lidar_data.timestamp}, Points: {len(lidar_data)}")

    unique_tags = set()
    for detection in lidar_data:
        unique_tags.add(detection.object_tag)
        if detection.object_tag == 10:
             print(f"  Vehicle point: {detection.point}, Object Index: {detection.object_idx}")

    print(f"  Detected Semantic Tags: {unique_tags}")

def process_radar_data(radar_measurement):
    print(f"Radar Frame: {radar_measurement.frame}, Timestamp: {radar_measurement.timestamp}, Detections: {len(radar_measurement)}")
    for detection in radar_measurement:
        print(f"  Depth: {detection.depth:.2f}m, Azimuth: {np.degrees(detection.azimuth):.2f}deg, Alt: {np.degrees(detection.altitude):.2f}deg, Vel: {detection.velocity:.2f}m/s")
    

client = None
world = None
ego_vehicle = None
sensor_list = [] 

try:
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.find('vehicle.toyota.prius')
    if not vehicle_bp:
        print("Error: Toyota Prius blueprint not found!")
        exit()

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("Error: No spawn points found in the map!")
        exit()
    spawn_point = random.choice(spawn_points)

    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f'Spawned ego vehicle: {ego_vehicle.type_id} (ID: {ego_vehicle.id})')
    sensor_list.append(ego_vehicle) 

    ego_vehicle.set_autopilot(True)

    spectator = world.get_spectator()
    vehicle_transform = ego_vehicle.get_transform()
    spectator.set_transform(carla.Transform(
            vehicle_transform.location + carla.Location(z=30, x=-15), 
            carla.Rotation(pitch=-30, yaw=vehicle_transform.rotation.yaw)
        ))

    ss_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    ss_camera_bp.set_attribute('image_size_x', '800')
    ss_camera_bp.set_attribute('image_size_y', '600')
    ss_camera_bp.set_attribute('fov', '90')

    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '35') 
    radar_bp.set_attribute('vertical_fov', '20')  
    radar_bp.set_attribute('range', '100') 
    radar_bp.set_attribute('points_per_second', '10000') 

    ss_lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    ss_lidar_bp.set_attribute('range', '100')
    ss_lidar_bp.set_attribute('channels', '64')
    ss_lidar_bp.set_attribute('points_per_second', '1120000') 
    ss_lidar_bp.set_attribute('rotation_frequency', '10') 
    ss_lidar_bp.set_attribute('upper_fov', '10.0')
    ss_lidar_bp.set_attribute('lower_fov', '-30.0')
  
    camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
    radar_transform = carla.Transform(carla.Location(x=2.2, z=0.6))  
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))  

    ss_camera = world.spawn_actor(
        ss_camera_bp,
        camera_transform,
        attach_to=ego_vehicle)
    print(f'Spawned sensor: {ss_camera.type_id} (ID: {ss_camera.id})')
    sensor_list.append(ss_camera)

    radar = world.spawn_actor(
        radar_bp,
        radar_transform,
        attach_to=ego_vehicle)
    print(f'Spawned sensor: {radar.type_id} (ID: {radar.id})')
    sensor_list.append(radar)

    ss_lidar = world.spawn_actor(
        ss_lidar_bp,
        lidar_transform,
        attach_to=ego_vehicle)
    print(f'Spawned sensor: {ss_lidar.type_id} (ID: {ss_lidar.id})')
    sensor_list.append(ss_lidar)

    ss_camera.listen(lambda data: process_semantic_camera(data))
    radar.listen(lambda data: process_radar_data(data))
    ss_lidar.listen(lambda data: process_semantic_lidar(data))

    print("\nAll actors spawned and sensors listening. Press Ctrl+c or Ctrl+s to stop.")
    while True:
        world.wait_for_tick() 
        

except KeyboardInterrupt:
    print('\nSimulation stopped by user.')

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    print('Destroying actors...')
    if client: 
        for sensor in sensor_list:
            if hasattr(sensor, 'is_listening') and sensor.is_listening:
                 sensor.stop()
                 print(f'Stopped sensor: {sensor.type_id} (ID: {sensor.id})')

        if sensor_list: 
             print(f'Destroying {len(sensor_list)} actors.')
             client.apply_batch([carla.command.DestroyActor(actor) for actor in sensor_list if actor.is_alive])
             time.sleep(0.5)

    print('Actors and NPC destroyed.')
