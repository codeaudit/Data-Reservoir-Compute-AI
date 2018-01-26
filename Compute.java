
package data.reservoir.compute.ai;

public abstract class Compute implements java.io.Serializable {

    final Reservoir reservoir;

    public Compute(Reservoir r) {
        reservoir = r;
    }

    public void resetHeldState() {
    }

    public abstract void compute();

    public abstract int nGather();
    
    public abstract int nScatterGeneral();
    
//  Number of calls to Reservoir multiplyWithWeights or multiplyWithWeightAddTo    
    public abstract int nCompute();

    public abstract int buffersRequired();

}
