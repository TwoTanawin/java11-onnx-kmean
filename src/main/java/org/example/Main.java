package org.example;

import ai.onnxruntime.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.AbstractMap;
import java.util.Map;

public class Main {

    // unchanged: your inference for one session
    public static int assignNewPoint(float lat, float lon, OrtSession session, OrtEnvironment env) throws OrtException {
        float[] inputData = new float[]{lat, lon};
        long[] shape = new long[]{1, 2};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)) {
            String inputName = session.getInputNames().iterator().next();
            String outputName = session.getOutputNames().iterator().next();

            try (OrtSession.Result result = session.run(Map.of(inputName, inputTensor))) {
                OnnxValue val = result.get(outputName).orElseThrow();
                long clusterId = ((long[]) val.getValue())[0];
                return (int) clusterId;
            }
        }
    }

    // Reactive helper: load a session, run inference, then close the session automatically
    static Mono<Map.Entry<Integer, Integer>> runForRadius(
            OrtEnvironment env,
            String modelDir,
            Integer km,  // Changed to Integer instead of int
            float lat,
            float lon
    ) {
        return Mono.using(
                // resource supplier (create OrtSession)
                () -> {
                    String modelPath = modelDir + File.separator + "kmeans_" + km + "km_model.onnx";
                    OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                    return env.createSession(modelPath, opts);
                },
                // resource consumer (do the work with the session)
                session -> Mono.fromCallable(() -> assignNewPoint(lat, lon, session, env))
                        .subscribeOn(Schedulers.boundedElastic())
                        .map(clusterId -> new AbstractMap.SimpleEntry<>(km, clusterId)),
                // resource cleanup
                session -> {
                    try { session.close(); } catch (Exception ignore) {}
                }
        );
    }

    public static void main(String[] args) {
        String modelDir = "E:\\Hydroneo\\Analytics\\disease\\models";
        int[] radii = {10, 30, 50};

        float newLat = 6.6198218f;
        float newLon = 100.0785343f;
        System.out.printf("%nNew point: (%.6f, %.6f)%n%n", newLat, newLon);

        // Create the environment once, close it when the pipeline is done
        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {

            // Fan-out across radii in parallel, run all, collect, then print.
            Flux.fromStream(java.util.Arrays.stream(radii).boxed())
                    .flatMap(km -> runForRadius(env, modelDir, km, newLat, newLon), radii.length)
                    .sort(Map.Entry.comparingByKey())
                    .doOnNext(e -> System.out.printf("At %dkm radius → belongs to cluster %d%n", e.getKey(), e.getValue()))
                    .then()
                    .block();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}