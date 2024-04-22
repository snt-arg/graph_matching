# Graph Matching

## Overview

Graph Matching stores and performs graph operations related to S-Graphs.

### License

The source code is released under GPLv3 License [![License: MIT](https://img.shields.io/badge/License-GPLv2-yellow.svg)](https://opensource.org/license/gpl-3-0).

**Author: Jose Andres Millan Romera<br />
Affiliation: [University of Luxembourg](https://www.anybotics.com/)<br />
Maintainer: Jose Andres Millan Romera, josmilrom@gmail.com**

The graph_matching package has been tested under [ROS2] Humble on Ubuntu 20.04.
This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.


## Installation

### Installation from Packages

The only tested ROS version for this package is ROS2 Humble
    
#### Dependencies

	- pip install networkx[default]
	- pip install -U scikit-learn
	- CLIPPER https://github.com/mit-acl/clipper

#### Building

To build from source, clone the latest version from this repository into your catkin workspace and compile the package using

	cd catkin_workspace/src
	git clone https://github.com/snt-arg/graph_matching.git
	cd ../
	colcon build


### Unit Tests (TODO)

Run the unit tests with

	catkin_make run_tests_ros_package_template

## Usage

Run the main node with

	ros2 launch graph_matching graph_matching.launch.py

Run the main and tester node with

	ros2 launch graph_matching graph_matching_tester.launch.py 

## Config files (TODO)

Config file folder/set 1

* **config_file_1.yaml** Shortly explain the content of this config file

Config file folder/set 2

* **...**

## Launch files

* **graph_matching.launch.py :** Launch of graph matching node

* **graph_matching_tester.launch.py :** Launch of graph matching node and, a second later, the tester node


## Nodes

### ros_package_template

Stores and performs graph operations related to S-Graphs.


#### Subscribed Topics

* **`/graphs`** ([graph_matching_msgs/Graph])


#### Published Topics

* **`/best_match`** ([graph_matching_msgs/Match])
* **`/unique_match`** ([graph_matching_msgs/Match])
* **`/best_match_visualization`** ([visualization_msgs/MarkerArray)
* **`/visualization_match_visualization`** ([visualization_msgs/MarkerArray)

#### Services

* **`subgraph_match_srv`** ([graph_matching_msgs/SubgraphMatchSrv])

	Returns the subgraph match (list of pairs of nodes) most probable, subject to matching type. NOT SUPPORTED AT THE MOMENT

		ros2 service call /subgraph_match_srv


#### Parameters (TODO)

* **`subscriber_topic`** (string, default: "/temperature")

	The name of the input topic.

* **`cache_size`** (int, default: 200, min: 0, max: 1000)

	The size of the cache.




