maps_config = {
    "race1": {
        "path": './sim_world/race_tracks/1.PNG',
        "start_coordinates": (90, 550),
        "check_point_list": [(290, 550), (670, 250), (1210, 160)]
    },
    "race2": {
        "path": './sim_world/race_tracks/2.PNG',
        "start_coordinates": (65, 560),
        "check_point_list": [(1060, 560), (640, 360), (250, 150), (1200, 150)]
    },
    "race3": {
        "path": './sim_world/race_tracks/3.PNG',
        "start_coordinates": (65, 560),
        "check_point_list": [(280,380), (375, 160), (790, 160), (1200, 160)]
    },
    "race4": {
        "path": './sim_world/race_tracks/4.PNG',
        "start_coordinates": (65, 560),
        "check_point_list": [(680, 570), (1090, 325), (80, 315), (1200, 150)]
    },
    "open3_half": {
        "path": './sim_world/open_world/3.PNG',
        "start_coordinates": (52, 180),
        "check_point_list": [(70, 260), (280, 360), (800, 160) , (1200, 160)]
    },
    "open3": {
        "path": './sim_world/open_world/3.PNG',
        "start_coordinates": (52, 180),
        "check_point_list": [(70, 260), (280, 360), (800, 160) , (1200, 160), (925, 560), (500, 560), (60, 560)]
    },
}

def get_map_config(map_name):
    return maps_config.get(map_name, None)
