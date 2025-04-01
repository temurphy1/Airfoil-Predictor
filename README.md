# Airfoil-Predictor
A neural network that's trained off of aero data from NACA airfoils and can predict aero data of new NACA airfoils

We're expecting data in the format of something like this:

NACA Airfoil Number | C_lift | C_drag | C_moment 

0012                |  .4    | .06     | .25
....                |   ...  |  ....    |...


We're going to pick the flow conditions such that we're not crossing between flow regimes, aka the range of the flow conditions will be set such that
the patterns between the flow conditions and the aero data will be as simple as possible.

The idea is that we'll handle this data & shove it into a NN, and ideally it'd be able to start *predicting* the aero data for NACA airfoils it's never seen before.
Note that NACA airfoils are identified by their NACA airfoil number, which is derived based on key geometric characteristics of the airfoil. 

Once this is done we can start expandng our inputs, so that instead of just putting in the NACA Airfoil number, we can put angle_of_attack, air velocity, etc. etc.

