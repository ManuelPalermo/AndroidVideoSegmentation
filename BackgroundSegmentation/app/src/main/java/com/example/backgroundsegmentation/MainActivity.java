package com.example.backgroundsegmentation;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.SystemClock;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    JavaCameraView cameraHandler;
    SegmentationModel segModel;

    FrameLayout cameraView;
    TextView inferenceView;
    TextView classesView;

    int cameraWidth;
    int cameraHeight;

    Mat inframe;
    Mat modelMat;
    Mat upModelMat;
    Mat resized;
    Mat outputFrame;
    ArrayList<String> classesFound;

    // ---------------------------------  load opencv library --------------------------------------
    static {
        if (!OpenCVLoader.initDebug()) ;// Handle initialization error
        else System.loadLibrary("opencv_java3");
    }

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    cameraHandler.enableView(); break;
                default:
                    super.onManagerConnected(status); break;
            }
        }
    };
    // ---------------------------------------------------------------------------------------------

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // get interface objects
        cameraView = findViewById(R.id.cameraLayout);
        classesView = findViewById(R.id.classesFound);
        inferenceView = findViewById(R.id.inferenceTime);

        // create camera handler
        cameraHandler = new JavaCameraView(this, 0);
        cameraHandler.setCvCameraViewListener(this);
        cameraHandler.setVisibility(SurfaceView.VISIBLE);
        cameraHandler.setLayoutParams(new FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
        cameraView.addView(cameraHandler);
        cameraHandler.enableView();

        // create tensorflow model
        try {segModel = new SegmentationModel(MainActivity.this);}
        catch (IOException e) {e.printStackTrace();}

        // ask for permissions
        askForPermission(Manifest.permission.CAMERA, 10);                   // ask camera permission
        askForPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE, 10);   // ask storage permission
        askForPermission(Manifest.permission.READ_EXTERNAL_STORAGE, 10);    // ask storage permission
    }

    private void askForPermission(String permission, Integer requestCode) {
        if (ContextCompat.checkSelfPermission(MainActivity.this, permission) != PackageManager.PERMISSION_GRANTED) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this, permission)) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
            } else {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
            }
        }
    }

    @Override
    public void onResume(){
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseLoaderCallback);
        } else {
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        try {segModel = new SegmentationModel(MainActivity.this);}
        catch (IOException e) {e.printStackTrace();}
    }

    @Override
    public void onPause() {
        super.onPause();
        if (cameraHandler != null)
            cameraHandler.disableView();
        if(segModel!=null) {
            segModel.close();
        }
    }

    @Override
    protected void onDestroy() {
        //stop camera
        super.onDestroy();
        if(cameraHandler!=null) {
            cameraHandler.disableView();
        }
        if(segModel!=null) {
            segModel.close();
        }
    }

    public void onCameraViewStarted(int width, int height){
        cameraWidth = width;
        cameraHeight = height;

        inframe = new Mat(cameraHeight, cameraWidth, CvType.CV_8UC3);
        modelMat = new Mat(256, 256, CvType.CV_8UC3);
        upModelMat = new Mat(cameraHeight, cameraHeight, CvType.CV_8UC3);
        resized = new Mat(cameraHeight, cameraWidth, CvType.CV_8UC3);
        outputFrame = new Mat(cameraHeight, cameraWidth, CvType.CV_8UC4);
        classesFound = new ArrayList<>();
}

    public void onCameraViewStopped(){
        Log.i("onCameraViewStopped: " ,"Camera view has stopped, releasing resources");
        inframe.release();
        modelMat.release();
        upModelMat.release();
        resized.release();
        outputFrame.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame){
        final long startTime = SystemClock.uptimeMillis();
        // convert to rgb image
        Imgproc.cvtColor(inputFrame.rgba(), inframe, Imgproc.COLOR_RGBA2RGB);

        // crop image(to square shape) and downsize image
        Imgproc.resize(new Mat(inframe, new Rect((int)(cameraWidth*0.25), 0, cameraHeight, cameraHeight)),
                modelMat, modelMat.size(), 0, 0, Imgproc.INTER_LANCZOS4);

        // segment image
        classesFound.clear();
        segModel.segmentImage(modelMat, classesFound);

        // return segmentation to original image size
        Imgproc.resize(modelMat, upModelMat, upModelMat.size(), 0, 0, Imgproc.INTER_CUBIC);

        // reshape image to original shape
        Core.copyMakeBorder(upModelMat, resized,
                    0, 0, (cameraWidth-cameraHeight)/2, (cameraWidth-cameraHeight)/2,
                    Core.BORDER_CONSTANT, new Scalar(0));

        // blend segmentation with camera frame
        Core.addWeighted(inframe, 0.5, resized, 0.5, 0.0, outputFrame);

        // update text
        runOnUiThread(new Runnable() {
            public void run() {
                inferenceView.setText("Inference(ms): " + (SystemClock.uptimeMillis() - startTime));
                classesView.setText("Classes: " + classesFound.toString());
            }
        });
        return outputFrame;
    }
}
