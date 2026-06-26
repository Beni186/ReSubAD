import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: WebAD - Web-based Anomaly Detection Dataset'''

# WebAD 有很多類別，這裡列出所有的類別
WEBAD_CLS_NAMES = [
    'abacus', 'air_conditioner', 'airplane', 'alarm_clock', 'ambulance',
    'apple', 'atm', 'backpack', 'badge', 'banana', 'bandage', 'barbell',
    'barometer', 'barricade', 'baseball_bat', 'baseball_glove', 'basketball',
    'bathtub', 'battery', 'beaker', 'bed', 'belt', 'bench', 'bicycle',
    'bicycle_helmet', 'binoculars', 'blazer', 'blender', 'blouse', 'boat',
    'book', 'bookcase', 'bookshelf', 'boot', 'boots', 'bottle', 'bowl',
    'box', 'bracelet', 'bridge', 'briefcase', 'broccoli', 'broom', 'bucket',
    'bulldozer', 'Bunsen_burner', 'bus', 'bus_stop', 'cabinet', 'cable',
    'cake', 'calculator', 'calendar', 'camera', 'camera_tripod', 'camping_stove',
    'camping_tent', 'can', 'candle', 'cap', 'car', 'carbon_monoxide_detector',
    'carpet', 'carrot', 'cash_register', 'cell_phone', 'chair', 'chalkboard',
    'charger', 'chest', 'clipboard', 'clock', 'closet', 'clothes', 'coat',
    'coffee_machine', 'coffee_maker', 'compass', 'computer', 'crane',
    'credit_card', 'crosswalk', 'crosswalk_sign', 'crutches', 'cup', 'curtains',
    'delivery_van', 'desk', 'dining_table', 'dishwasher', 'dj_turntable',
    'document', 'donut', 'door', 'doorbell', 'drawer', 'dress', 'drill',
    'drone', 'drum_set', 'dryer', 'dumbbell', 'dump_truck', 'dvd_player',
    'earbuds', 'ear_protection', 'earrings', 'elevator', 'envelope',
    'escalator', 'excavator', 'exercise_bike', 'extension_cord', 'face_mask',
    'fan', 'faucet', 'fence', 'file', 'fire_extinguisher', 'fire_hose',
    'fire_hydrant', 'fire_truck', 'first_aid_kit', 'fishing_rod', 'flash_drive',
    'flashlight', 'flask', 'folder', 'football', 'fork', 'forklift', 'freezer',
    'frisbee', 'furniture', 'game_controller', 'garage_door_opener',
    'garbage_truck', 'garden_shears', 'gate', 'glasses', 'globe', 'gloves',
    'glue', 'goggles', 'golf_club', 'graduated_cylinder', 'guitar',
    'gym_equipment', 'hair_dryer', 'hammer', 'handbag', 'hanger', 'hard_drive',
    'hard_hat', 'hat', 'headphones', 'headset', 'heater', 'helicopter',
    'helmet', 'highlighter', 'hiking_boots', 'hockey_stick', 'hoodie', 'hose',
    'hospital_bed', 'hot_dog', 'hygrometer', 'id_card', 'iron', 'IV_stand',
    'jacket', 'jar', 'jeans', 'joystick', 'jump_rope', 'karaoke_machine',
    'kettle', 'kettlebell', 'keyboard', 'kiosk', 'kite', 'knife', 'ladder',
    'lamp', 'lantern', 'laptop', 'lawn_mower', 'leaf_blower', 'level',
    'lifebuoy', 'life_jacket', 'light_bulb', 'lighter', 'light_switch',
    'loading_dock', 'luggage', 'magazine', 'mailbox', 'map', 'marker',
    'matches', 'medical_mask', 'medicine_ball', 'megaphone', 'microphone',
    'microscope', 'microwave', 'minivan', 'mirror', 'mittens', 'modem',
    'monitor', 'mop', 'motorcycle', 'mouse', 'necklace', 'newspaper',
    'notebook', 'orange', 'outlet', 'oven', 'package', 'paint_brush',
    'paintbrush', 'pan', 'pants', 'paper_clip', 'parking_meter',
    'passenger_seat', 'pen', 'person', 'petri_dish', 'phone', 'piano',
    'pill_bottle', 'pipette', 'pizza', 'plate', 'pliers', 'plug', 'plunger',
    'police_car', 'pot', 'potted_plant', 'power_strip', 'printer', 'projector',
    'punching_bag', 'purse', 'raincoat', 'rake', 'razor', 'recycling_bin',
    'refrigerator', 'remote', 'remote_control', 'ring', 'roller',
    'roller_brush', 'roller_skates', 'router', 'rug', 'ruler', 'safety_vest',
    'sandals', 'sandwich', 'saw', 'scarf', 'school_desk', 'scissors',
    'scooter', 'screwdriver', 'sd_card', 'seat_belt', 'security_camera',
    'shelf', 'ship', 'shirt', 'shoe', 'shoes', 'shopping_bag', 'shopping_cart',
    'shorts', 'shovel', 'shower', 'sink', 'skateboard', 'ski', 'skirt',
    'skis', 'sleeping_bag', 'slippers', 'smartphone', 'smoke_detector',
    'snowboard', 'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa',
    'speaker', 'spoon', 'sports_ball', 'sports_car', 'sprinkler', 'staircase',
    'stairs', 'stapler', 'stethoscope', 'stool', 'stop_sign', 'stove',
    'street_lamp', 'street_light', 'street_sweeper', 'stretcher', 'subway',
    'suit', 'suitcase', 'sunglasses', 'surfboard', 'surgical_tools',
    'surveillance_camera', 'sweater', 'syringe', 'table', 'tablet',
    'tank_top', 'tape', 'tape_dispenser', 'tape_measure', 'taxi',
    'teddy_bear', 'telephone', 'telescope', 'television', 'tennis_racket',
    'tent', 'test_tube', 'thermometer', 'thermos', 'thermostat', 'tie',
    'toaster', 'toilet', 'toolbox', 'toothbrush', 'towel', 'tractor',
    'traffic_cone', 'traffic_light', 'trailer', 'train', 'trash_can',
    'treadmill', 'tripod', 'trolley', 'truck', 't-shirt', 'tunnel', 'tv',
    'umbrella', 'usb_cable', 'vacuum_cleaner', 'vase', 'vending_machine',
    'violin', 'wallet', 'wardrobe', 'washing_machine', 'watch',
    'water_fountain', 'watering_can', 'wheelbarrow', 'wheelchair', 'whistle',
    'whiteboard', 'window', 'wine_glass', 'wrench', 'yoga_mat'
]

WEBAD_ROOT = os.path.join(DATA_ROOT, 'webad')

class WebADDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=None, aug_rate=0.0, root=WEBAD_ROOT, training=True, split=None):
        # 如果沒有指定類別名稱，則動態獲取
        if clsnames is None:
            clsnames = self._get_available_classes(root)
        
        super(WebADDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training,
            split=split
        )
    
    def _get_available_classes(self, root):
        """動態獲取實際存在的類別名稱"""
        normal_dir = os.path.join(root, 'normal_images')
        anomaly_dir = os.path.join(root, 'anomaly_images')
        
        classes = set()
        
        if os.path.exists(normal_dir):
            classes.update([d for d in os.listdir(normal_dir) 
                          if os.path.isdir(os.path.join(normal_dir, d))])
        
        if os.path.exists(anomaly_dir):
            classes.update([d for d in os.listdir(anomaly_dir) 
                          if os.path.isdir(os.path.join(anomaly_dir, d))])
        
        return sorted(list(classes))