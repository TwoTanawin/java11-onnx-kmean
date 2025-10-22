package org.example;

import ai.onnxruntime.*;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;

public class Main {
    public static int assignNewPoint(float lat, float lon, OrtSession session, OrtEnvironment env) throws OrtException {
        // ONNX expects float32 [[lat, lon]]
        float[] inputData = new float[]{lat, lon};
        long[] shape = new long[]{1, 2}; // 1 row, 2 columns

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)) {
            String inputName = session.getInputNames().iterator().next();
            String outputName = session.getOutputNames().iterator().next();

            try (OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor))) {
                OnnxValue val = result.get(outputName).orElseThrow();
                long clusterId = ((long[]) val.getValue())[0];
                return (int) clusterId;
            }
        }
    }


    public static void main(String[] args) {
        String modelDir = "E:\\Hydroneo\\Analytics\\disease\\models";
        int[] radii = {10, 30, 50};

        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            Map<Integer, OrtSession> sessions = new HashMap<>();

            // Load sessions
            for (int km : radii) {
                String modelPath = modelDir + File.separator + "kmeans_" + km + "km_model.onnx";
                OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                OrtSession session = env.createSession(modelPath, options);
                sessions.put(km, session);
            }

            // Example new coordinates
            float newLat = 6.6198218f;
            float newLon = 100.0785343f;
            System.out.printf("%n New point: (%.6f, %.6f)%n%n", newLat, newLon);

            // Run inference for each radius
            for (int km : radii) {
                OrtSession session = sessions.get(km);
                int clusterId = assignNewPoint(newLat, newLon, session, env);
                System.out.printf("At %dkm radius → belongs to cluster %d%n", km, clusterId);
            }

            // Cleanup
            for (OrtSession s : sessions.values()) {
                s.close();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}