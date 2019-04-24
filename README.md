# Random Waypoint Model

Description
-----------
This is a python 3 code library for simulation of mobility models. Generate random way point model for sensor network[[1]](#references).

Dependencies
------------
NumPy and Matplotlib

Examples
--------
### Mobility Models
For example, to create a Random Waypoint model instace, use the following commands:
```python
>>> rwp = random_waypoint(10, dimensions=(100, 100), velocity=(0.01, 2), wt_max=0)
```
This will create a Random Waypoint instance with 10 nodes in a simulation area of 100x100 units, 
velocity chosen from a uniform distribution between 0.01 and 2 units/step
and maximum waiting time of 0.0 steps.
This object is a generator that yields the position of the nodes in each step.
For example, to print a 2-dimensional array with the node positions produced in the first step, call
```python
>>> positions = next(rwp)
>>> print positions
```
You can also iterate over the positions produced in each step:
```python
>>> for positions in rwp:
...     print positions
... 
```
Result
--------
The trajectory of the sensor nodes is shown below:
![Alt text]( results.png?raw=true "")<br />

References
----------
[1] Available online at: https://github.com/panisson/pymobility
