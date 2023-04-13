import React, { useRef, useState } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [squatCount, setSquatCount] = useState(0);

  // Load posenet
  const runPosenet = async () => {
    const net = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: 320, height: 240 },
      multiplier: 0.5,
      scale: 0.8,
    });

    setInterval(() => {
      detect(net);
    }, 100);
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get video properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Make pose estimation
      const pose = await net.estimateSinglePose(video);

      // Determine squat position
      const leftHip = pose.keypoints[11];
      const rightHip = pose.keypoints[12];
      const leftKnee = pose.keypoints[13];
      const rightKnee = pose.keypoints[14];
      const leftAnkle = pose.keypoints[15];
      const rightAnkle = pose.keypoints[16];

      // Determine squat depth
      const squatDepth = Math.min(
        leftHip.score,
        rightHip.score,
        leftKnee.score,
        rightKnee.score,
        leftAnkle.score,
        rightAnkle.score
      );

      // Draw keypoints and skeleton on canvas
      drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);

      // Check if squat is detected
      if (squatDepth >= 0.50) {
        setSquatCount((prevCount) => prevCount + 1);
      }
    }
  };

  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas) => {
    const ctx = canvas.current.getContext("2d");
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;

    drawKeypoints(pose.keypoints, 0.6, ctx);
    drawSkeleton(pose.keypoints, 0.7, ctx);
  };

  runPosenet();

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />

        <div
          style={{
            position: "absolute",
            top: "10px",
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 10,
            color: "#fff",
            fontSize: "3em",
            fontFamily: "sans-serif",
            fontWeight: "bold",
            textShadow: "2px 2px #000",
          }}
        >
          Squats: {squatCount}
        </div>
      </header>
    </div>
  );
}

export default App;

