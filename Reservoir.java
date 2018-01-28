package data.reservoir.compute.ai;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

public class Reservoir implements Serializable {

    private final static float MIN_SQ = 1e-20f;
    private final int computeSize;
    private final int inputBlocks;
    private final int writeBlocks;
    private final int generalBlocks;
    private final int outputSize;

    private int hashIndex;
    private int weightIndex;

    private int weightSize;
    private float[] weights;
    private transient float[] dataReservoir;
    private transient float[][] computeBuffers;
    private transient RNG rng;
    private final ArrayList<Compute> list; // list of all compute units for the AI
    
//  The data resevoir is in 3 parts [input][write][general]    
//  All calulations are done on vectors of computeSize (= one block.)
//  computeSize must be an integer power of 2 >=16 (eg. 16,32,64,128...)    
//  Subclasses of Compute act on the data reservoir.  Information from all
//  the data reservoir is selected by weighting and random projection and 
//  sent to a Compute subclass, the result is generally scattered to the
//  general part again by random projection and weight based select.
//  This allows complex interconnection and module behavior to evolve.
//  Information like bias terms and random numbers go to write section.
//  The output size can be arbitary.
//  After adding all the Compute subclasses you want you should call
//  prepareForUse().  
//  Generally you should call clearReservoir(), setInput(...), computeAll()
//  getOutput(...) however depending on the system you are trying to create
//  you can do other things.    
    public Reservoir(int computeSize, int inputBlocks, int writeBlocks, int generalBlocks, int outputSize) {
        if (computeSize < 16 | (computeSize & computeSize - 1) != 0) {
            throw new IllegalArgumentException("computeSize must be a power of 2 and at least 16");
        }
        this.computeSize = computeSize;
        this.inputBlocks = inputBlocks;
        this.writeBlocks = writeBlocks;
        this.generalBlocks = generalBlocks;
        this.outputSize = outputSize;
        list = new ArrayList<>();
    }

//  Add an object that is a subclass of Compute    
    public void addComputeUnit(Compute c) {
        list.add(c);
    }

// Call after adding compute units. Don't add more compute units and call again.
// Also sets up after deserialization. Called from the readObject() method.    
    public void prepareForUse() {
        int rTotal = computeSize * (inputBlocks + writeBlocks + generalBlocks);
        int rGen = computeSize * generalBlocks;
        dataReservoir = new float[rTotal];  //transient
        rng = new RNG();    //transient
        int nBuffers = 0;
        weightSize = 0;
        for (Compute c : list) {
            weightSize += c.nGather() * rTotal + c.nScatterGeneral() * rGen + c.nCompute() * computeSize;
            if (nBuffers < c.buffersRequired()) {
                nBuffers = c.buffersRequired();
            }
        }
        if (weights == null) {
            weights = new float[weightSize];
            for (int i = 0; i < weightSize; i++) {
                weights[i] = rng.nextFloatSym();
            }
        }
        computeBuffers = new float[nBuffers][computeSize];
    }

// Compute subclass object act on the data reservoir.    
    public void computeAll() {
        hashIndex = 0;
        weightIndex = 0;
        for (Compute c : list) {
            c.compute();
        }
   //     if(weightIndex!=weightSize) System.out.println("Error");
    }
    
//  reset internal state of data reservoir.
    public void clearReservoir(){
        Arrays.fill(dataReservoir, 0f);
    }
    
// clears all held state such as in associative memory.    
    public void clearHeldStateAll() {
        for (Compute c : list) {
            c.resetHeldState();
        }
    }
//  Sets the input part of the reservoir.  Must be computeSize*inputBlocks length.
    public void setInput(float[] input) {
        System.arraycopy(input, 0, dataReservoir, 0, computeSize * inputBlocks);
    }

//  Gets the outputs from the general part of data reservoir.    
    public void getOutput(float[] output) {
        System.arraycopy(dataReservoir, computeSize * (inputBlocks + writeBlocks), output, 0, outputSize);
    }

//  Turns the Reservoir object into its own descendent for evolution.
//  You should preserve the prior state by calling getWeigths(...) first.
    public void mutate(long mutatePrecision) {
        for (int i = 0; i < weightSize; i++) {
            weights[i] = rng.mutateXSym(weights[i], mutatePrecision);
        }
    }

    public int getWeightSize() {
        return weightSize;
    }

//  Gets the current state of the Reservoir object, excluding any state held in 
//  Compute subclasses such as associative memory. 
    public void getWeights(float[] vec) {
        System.arraycopy(weights, 0, vec, 0, weightSize);
    }

//  Allows you to put back the original state if a evolution step fails to 
//  improve behavior on a task.    
    public void setWeights(float[] vec) {
        System.arraycopy(vec, 0, weights, 0, weightSize);
    }
    
//  all processing is done on blocks of computeSize.
    public int getComputeSize(){
        return computeSize;
    }

//  Allows access to internal weights from Compute subclasses.  
    void multiplyWithWeights(float[] resultVec, float[] x) {
        for (int i = 0; i < computeSize; i++) {
            resultVec[i] = x[i] * weights[weightIndex++];
        }
    }

//  Allows access to internal weights from Compute subclasses. 
    void multiplyWithWeightsAddTo(float[] resultVec, float[] x) {
        for (int i = 0; i < computeSize; i++) {
            resultVec[i] += x[i] * weights[weightIndex++];
        }
    }

//  Fast random projection (random dot product) with the hashing value index
//  maintained by the Reservoir object.    
    void randomProjection(float[] x) {
        WHT.fastRP(x, hashIndex++);
    }

//  Gathers data from all the data reservoir down to vector of computeSize for
//  an object of subclass Compute to act on.    
    void gather(float[] g) {
        int totalBlocks = inputBlocks + writeBlocks + generalBlocks;
        int rIndex = 0;
        Arrays.fill(g, 0f);
        for (int i = 0; i < totalBlocks; i++) {
            for (int j = 0; j < computeSize; j++) {
                float p = weights[weightIndex++];
                p *= p;       // Bias toward not selecting
                g[j] += dataReservoir[rIndex++] * p;
            }
            randomProjection(g);
        }
    }

//  Scatter the output of a Compute subclass object to the general part of the
//  data reservoir via a blending process.    
//  s is destroyed by this method, it is assumed the compute unit will not need
//  it again.
    void scatterGeneral(float[] s) {
        int rIndex = computeSize * (inputBlocks + writeBlocks);
        for (int i = 0; i < generalBlocks; i++) {
            randomProjection(s);
            for (int j = 0; j < computeSize; j++) {
                float p = weights[weightIndex++];
                p *= p;          //bias toward not selecting
                dataReservoir[rIndex] = p * s[j] + (1f - p) * dataReservoir[rIndex];
                rIndex++;
            }
        }
    }

//  Put data such as bias terms into the write part of data reservoir.    
    void scatterWrite(float[] s, int writeLocation) {
        System.arraycopy(s, 0, dataReservoir, computeSize * (inputBlocks + writeLocation), computeSize);
    }

//  Give the data in input part of the data reservoir a constant vector length.    
    void normalizeInput() {
        float sumSq = 0f;
        int inputLen = computeSize * inputBlocks;
        for (int i = 0; i < inputLen; i++) {
            sumSq += dataReservoir[i] * dataReservoir[i];
        }
        float adj = 1f / (float) Math.sqrt((sumSq / inputLen) + MIN_SQ);
        for (int i = 0; i < inputLen; i++) {
            dataReservoir[i] *= adj;
        }
    }

//  Buffers of computeSize for Compute subclass objects to use as needed.    
    float[] getComputeBuffer(int index) {
        return computeBuffers[index];
    }

//  Java deserialization    
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        prepareForUse(); //Set up all the buffers and working arrays
    }
}
