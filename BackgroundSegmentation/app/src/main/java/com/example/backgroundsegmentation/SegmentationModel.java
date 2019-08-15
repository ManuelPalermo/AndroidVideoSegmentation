package com.example.backgroundsegmentation;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.core.Scalar;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

public class SegmentationModel {
    private static final String MODEL_PATH = "mobilenet_v2_deeplab_v3_256_quant.tflite";    // model to use

    // image buffers shape
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 1;
    private static final int DIM_HEIGHT = 256;
    private static final int DIM_WIDTH = 256;
    private static final int INCHANNELS = 3;
    private static final int OUTCHANNELS = 1;

    private static final boolean useGpu = false;        // cant use gpu with quantized model(uint8 input)
    private GpuDelegate gpudelegate;
    private Interpreter tflite;

    private ByteBuffer inpImg;                          // model input buffer(uint8)
    private long[][] outImg;                            // model output buffer(int64)

    // classes to be displayed/colors and their respective names
    private int[][] colors = new int[21][3];
    private boolean[] displayClass = new boolean[21];
    private String[] classNames = new String[21];

    public SegmentationModel(Activity activity) throws IOException {
        try {
            gpudelegate = new GpuDelegate();
            Interpreter.Options options = (new Interpreter.Options()).addDelegate(gpudelegate);
            if (useGpu){ tflite = new Interpreter(loadModelFile(activity, MODEL_PATH), options);}
            else {tflite = new Interpreter(loadModelFile(activity, MODEL_PATH));}
            tflite.setNumThreads(4);
        }
        catch (IOException e) {e.printStackTrace();}

        inpImg = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_HEIGHT * DIM_WIDTH * DIM_PIXEL_SIZE * INCHANNELS);
        inpImg.order(ByteOrder.nativeOrder());

        outImg = new long[DIM_BATCH_SIZE][DIM_HEIGHT * DIM_WIDTH * OUTCHANNELS];

        colors[0]  = new int[]{0,   0,   0  };  displayClass[0]  = false; classNames[0]  = "bg";
        colors[1]  = new int[]{128, 0,   0  };  displayClass[1]  = false; classNames[1]  = "aeroplane";
        colors[2]  = new int[]{0,   128, 0  };  displayClass[2]  = true;  classNames[2]  = "bicycle";
        colors[3]  = new int[]{128, 128, 0  };  displayClass[3]  = false; classNames[3]  = "bird";
        colors[4]  = new int[]{0,   0,   128};  displayClass[4]  = false; classNames[4]  = "boat";
        colors[5]  = new int[]{128, 0,   128};  displayClass[5]  = true;  classNames[5]  = "bottle";
        colors[6]  = new int[]{0,   128, 128};  displayClass[6]  = true;  classNames[6]  = "bus";
        colors[7]  = new int[]{128, 128, 128};  displayClass[7]  = true;  classNames[7]  = "car";
        colors[8]  = new int[]{64,  0,   0  };  displayClass[8]  = true;  classNames[8]  = "cat";
        colors[9]  = new int[]{192, 0,   0  };  displayClass[9]  = true;  classNames[9]  = "chair";
        colors[10] = new int[]{64,  128, 0  };  displayClass[10] = false; classNames[10] = "cow";
        colors[11] = new int[]{192, 128, 0  };  displayClass[11] = true;  classNames[11] = "table";
        colors[12] = new int[]{64,  0,   128};  displayClass[12] = true;  classNames[12] = "dog";
        colors[13] = new int[]{192, 0,   128};  displayClass[13] = false; classNames[13] = "horse";
        colors[14] = new int[]{64,  128, 128};  displayClass[14] = true;  classNames[14] = "motorbike";
        colors[15] = new int[]{192, 128, 128};  displayClass[15] = true;  classNames[15] = "person";
        colors[16] = new int[]{0,   64,  0  };  displayClass[16] = true;  classNames[16] = "plant";
        colors[17] = new int[]{128, 64,  0  };  displayClass[17] = false;  classNames[17] = "sheep";
        colors[18] = new int[]{0,   192, 0  };  displayClass[18] = true;  classNames[18] = "sofa";
        colors[19] = new int[]{128, 192, 0  };  displayClass[19] = false;  classNames[19] = "train";
        colors[20] = new int[]{0,   64,  128};  displayClass[20] = true;  classNames[20] = "tv/monitor";
    }


    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void segmentImage(Mat modelMat, ArrayList<String> classesDetected) {
        // segment Mat inplace
        if(tflite!=null) {
            loadMatToBuffer(modelMat);
            tflite.run(inpImg, outImg);
            loadBufferToMat(modelMat, classesDetected);
        }
    }

    private void loadMatToBuffer(Mat inMat) {
        //convert opencv mat to tensorflowlite input
        inpImg.rewind();
        byte[] data = new byte[DIM_WIDTH * DIM_HEIGHT * INCHANNELS];
        inMat.get(0, 0, data);
        inpImg = ByteBuffer.wrap(data);
    }

    private void loadBufferToMat(Mat modelMat, ArrayList<String> classesDetected) {
        //convert tensorflowlite output to opencv mat
        boolean[] classesFound = new boolean[21];                               // temp bollean mask over calsses found
        Mat temp_outSegment = new Mat(DIM_HEIGHT, DIM_WIDTH, CvType.CV_32SC3);  // temp mask(Mat) -> class colors(int32)

        // major bottleneck(remove loop - load buffer directly to Mat somehow)
        Arrays.fill(classesFound, false);
        temp_outSegment.setTo(new Scalar(colors[0][0],colors[0][1],colors[0][2]));
        for(int y = 0; y < DIM_HEIGHT; y++) {
            for(int x = 0; x < DIM_WIDTH; x++) {
                int cl = (int)outImg[0][y * DIM_HEIGHT + x];
                if (displayClass[cl]){
                    temp_outSegment.put(y, x, colors[cl]);
                    classesFound[cl]=true;
                }
            }
        }
        temp_outSegment.convertTo(modelMat, CvType.CV_8UC3);

        for (int c = 0; c < classNames.length; c++) {
            if (classesFound[c]) classesDetected.add(classNames[c]);
        }
    }

    public void close() {
        if (tflite!=null) {
            tflite.close();
            tflite = null;
            gpudelegate.close();
        }
    }
}
