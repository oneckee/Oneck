./
├── config # [Config] configuration files for autonomous driving
│ ├── map # desired trajectory data
│ ├── config.json # configuration parameters file
│ └── config.py # load and apply the config.json
├── control # [Control] vehicle control
│ ├── control_input.py # convert and apply to the actual control input
│ ├── pid.py # calculate longitudinal control input with PID
│ └── pure_pursuit.py # calculate lateral control input with pure pursuit
├── localization # [Localization] vehicle position
│ └── path_manager.py # make local path from global path
├── mgeo # [HD Map] HD map processing
│ ├── lib
│ │ ├── mgeo # MGeo HD map loader repo (submodule)
│ │ └── mgeo_data # example drivable maps
│ ├── calc_mgeo_path.py # dijkstra path finder wrapper
│ ├── e_dijkstra.py # dijkstra algorithm
│ ├── get_mgeo.py # MGeo loader wrapper
│ └── mgeo_pub.py # ROS2 MGeo HD map publisher
├── perception # [Perception] object classification & filtering
│ ├── forward_object_detector.py # filter objects by driving path
│ └── object_info.py # classify object information
├── planning # [Planning] driving vehicle planning
│ └── adaptive_cruise_control.py # smart/adaptive cruise control planning
├── autonomous_driving.py # [Entry] autonomous driving example class
└── vehicle_state.py # [Status] vehicle status information
