// Subclasses implment neural network layers, associative memory and other
// useful computations on the data reservoir.
package data.reservoir.compute.ai;

public abstract class Compute implements java.io.Serializable {

    final Reservoir reservoir;

    public Compute(Reservoir r) {
        reservoir = r;
    }

 // override to clear any held state such as in associative memory.   
    public void resetHeldState() {
    }

//  Do the computation using the reservoir object for access.    
    public abstract void compute();

//  number of gather operations done.    
    public abstract int nGather();
    
//  number of scatter operations to the general section of the data reservoir.    
    public abstract int nScatterGeneral();
    
//  Number of calls to Reservoir multiplyWithWeights or multiplyWithWeightAddTo    
    public abstract int nCompute();

    public abstract int buffersRequired();

}
