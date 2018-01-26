# Data-Reservoir-Compute-AI
Neural network using a data reservoir to allow complicated interconnections and modules to evolve.
It uses a selective gather and scatter approach to pass small blocks of information to and from neural network layers, associative memory or any other type of compute. Gather and scatter is done using random projections (random dot products.) Each point in the data reservoir can create (if fully or partially selected) a unique pattern in the gathered vector.  Likewise unique patterns in the output of a neural network layer or other compute correspond to a unique point in the data reservoir which may be written there via a weight based blending process.
Anyway an evolution process does everything automatically.

The shared library is optional for Linux AMD64 and won't work (will throw an error) if you change the java package in the class files.
