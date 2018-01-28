package data.reservoir.compute.ai;

import java.util.Arrays;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

public class Main extends Application {

    static float[] input = new float[16];
    static float[] output = new float[3];

    public static void main(String[] args) {

        Reservoir r = new Reservoir(16, 1, 2, 2, 3);

        r.addComputeUnit(new ComputeNormalizeInput(r));
        r.addComputeUnit(new ComputeBiasWrite(r, 0));
        //r.addComputeUnit(new ComputeRndWrite(r, 1e-4f, 1));
        //r.addComputeUnit(new ComputeAMGT(r, 10,1f));// throw in to test 
        //r.addComputeUnit(new ComputeLayerSqHD(r, 3));
        r.addComputeUnit(new ComputeLayerReLU(r, 10));
        r.addComputeUnit(new ComputeLayerReLU(r, 10));
        r.prepareForUse();

        float[] parent = new float[r.getWeightSize()];
        for (int i = 0;
                i < 10000; i++) {
            float pCost = getCost(r);
            r.getWeights(parent);
            r.mutate(1000);
            float cCost = getCost(r);
            if (pCost < cCost) {
                r.setWeights(parent);
            } else {
                System.out.println(cCost);
            }

        }

        Arrays.fill(input,
                0f);
        input[0] = 1f;
        r.clearReservoir();

        r.setInput(input);

        r.computeAll();

        r.getOutput(output);

        System.out.println(Arrays.toString(output));

        input[1] = 1f;
        r.clearReservoir();

        r.setInput(input);

        r.computeAll();

        r.getOutput(output);

        System.out.println(Arrays.toString(output));

        input[2] = 1f;
        r.clearReservoir();

        r.setInput(input);

        r.computeAll();

        r.getOutput(output);

        System.out.println(Arrays.toString(output));

        launch(args);
    }

    public static float getCost(Reservoir r) {
        float cost = 0f;
        Arrays.fill(input, 0f);
        input[0] = 1f;
        r.clearReservoir();
        r.setInput(input);
        r.computeAll();
        r.getOutput(output);
        cost += (1f - output[0]) * (1f - output[0]);
        cost += output[1] * output[1] + output[2] * output[2];

        input[1] = 1f;
        r.clearReservoir();
        r.setInput(input);
        r.computeAll();
        r.getOutput(output);
        cost += (1f - output[1]) * (1f - output[1]);
        cost += output[0] * output[0] + output[2] * output[2];

        input[2] = 1f;
        r.clearReservoir();
        r.setInput(input);
        r.computeAll();
        r.getOutput(output);
        cost += (1f - output[2]) * (1f - output[2]);
        cost += output[0] * output[0] + output[1] * output[1];
        return cost;
    }

    @Override
    public void start(Stage primaryStage) {
        StackPane root = new StackPane();
        Scene scene = new Scene(root, 300, 250);
        primaryStage.setTitle("Test");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

}
