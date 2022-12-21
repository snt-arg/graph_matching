# Graph Manager

## Overview

Graph Manager stores and performs graph operations related to S-Graphs.

### License

The source code is released under MIT License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT).

**Author: Jose Andres Millan Romera<br />
Affiliation: [University of Luxembourg](https://www.anybotics.com/)<br />
Maintainer: Jose Andres Millan Romera, josmilrom@gmail.com**

The graph_manager package has been tested under [ROS2] Humble on Ubuntu 20.04.
This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.


## Installation

### Installation from Packages

The only tested ROS version for this package is ROS2 Humble
    
#### Dependencies

	- pip install networkx[default]

#### Building

To build from source, clone the latest version from this repository into your catkin workspace and compile the package using

	cd catkin_workspace/src
	git clone https://github.com/snt-arg/graph_manager.git
	cd ../
	colcon build


### Unit Tests (TODO)

Run the unit tests with

	catkin_make run_tests_ros_package_template

## Usage

Run the main node with

	ros2 launch graph_manager graph_manager.launch.py

Run the main and tester node with

	ros2 launch graph_manager graph_manager_tester.launch.py 

## Config files (TODO)

Config file folder/set 1

* **config_file_1.yaml** Shortly explain the content of this config file

Config file folder/set 2

* **...**

## Launch files

* **graph_manager.launch.py :** Launch of graph manager node

* **graph_manager_tester.launch.py :** Launch of graph manager node and, a second later, the tester node


## Nodes

### ros_package_template

Stores and performs graph operations related to S-Graphs.


#### Subscribed Topics

* **`/graphs`** ([graph_manager_msgs/Graph])


#### Published Topics (TODO)

...


#### Services

* **`subgraph_match_srv`** ([graph_manager_msgs/SubgraphMatchSrv])

	Returns the subgraph match (list of pairs of nodes) most probable, subject to matching type

		ros2 service call /subgraph_match_srv


#### Parameters (TODO)

* **`subscriber_topic`** (string, default: "/temperature")

	The name of the input topic.

* **`cache_size`** (int, default: 200, min: 0, max: 1000)

	The size of the cache.




