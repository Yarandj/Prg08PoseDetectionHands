// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import {HandLandmarker, FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import jsondata from './data.json'

with {type: "json"}; // Het pad naar je JSON-bestand
let handData = []
let posesArray = []

for (let i = 0; i < jsondata.poseList.length; i++) {

    if (jsondata.poseList[i].pose.length === 63) {

        handData.push(jsondata.poseList[i]);
    } else {
        console.log("data is niet de goede lengte" + jsondata.poseList[i].name + "nummer" + i);
    }

    // Doe iets met elk item in de JSON-data

}
console.log(jsondata.poseList.length)
console.log(handData)

const demosSection = document.getElementById("demos");
let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
let poseData = [];
// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2
    });
    demosSection.classList.remove("invisible");
};
createHandLandmarker();
/********************************************************************
 // Demo 2: Continuously grab image from webcam stream and detect it.
 ********************************************************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
// Check if webcam access is supported.
const hasGetUserMedia = () => {
    var _a;
    return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia);
};
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

function flattenData(data) {
    return data.reduce((acc, obj) => {
        return acc.concat(Object.values(obj));
    }, []);
}

// Define function to log hand coordinates
function logHandCoordinates(landmarks) {
    landmarks.forEach((landmark, index) => {
        console.log(`Hand ${index + 1} coordinates:`);
        landmark.forEach((point, pointIndex) => {
            console.log(`Point ${pointIndex + 1}: x=${point.x}, y=${point.y}, z=${point.z}`);
        });
    });
}

let training = []

// Function to simplify pose data and store it
function simplifyAndStorePose(landmarks, label) {
    const simplifiedLandmarks = [];
    landmarks.forEach(landmark => {
        simplifiedLandmarks.push(landmark.x, landmark.y, landmark.z);
    });
    poseData.push({pose: simplifiedLandmarks, label: label});
}

const testing = document.getElementById("old");
testing.addEventListener("click", logPoseData);

// Function to log pose data to console
function logPoseData() {
    poseData.forEach(pose => {
        console.log("Pose:", pose.pose);
        console.log("Label:", pose.label);
    });
}

// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let lastVideoTime = -1;
let results = undefined;
console.log(video);

async function predictWebcam() {
    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({runningMode: "VIDEO"});
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
        if (results.landmarks) {
            for (const landmarks of results.landmarks) {
                for (const point of landmarks) {
                    posesArray.push(point.x);
                    posesArray.push(point.y);
                    posesArray.push(point.z || 0); // Z-coördinaat, indien beschikbaar
                }
                training = flattenData(landmarks);
                // console.log(training)
            }
        }
    }
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 5
            });
            drawLandmarks(canvasCtx, landmarks, {color: "#FF0000", lineWidth: 2});
        }
    }
    // Check if results contain landmarks
    if (results && results.landmarks && results.landmarks.length > 0) {
        // Log hand coordinates
        // logHandCoordinates(results.landmarks);
        simplifyAndStorePose(results.landmarks[0], "rock");
    }
    canvasCtx.restore();
    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

const nn = ml5.neuralNetwork({task: 'classification', debug: true})

// vul hier zelf de rest van de data in
// ...
// nn.normalizeData()
//
// nn.train({ epochs: 10 }, () => finishedTraining())

console.log(handData.length)

// Functie om de data willekeurig te sorteren
handData = handData.sort(() => Math.random() - 0.5);

// Splitsen in train en testdata
const trainData = handData.slice(0, Math.floor(handData.length * 0.8)); // Voeg 1 toe om alle datapunten te hebben
const testData = handData.slice(Math.floor(handData.length * 0.8) + 1);

let correctPredictions = 0;

// Controleer of de data willekeurig is gesorteerd
console.log(handData);

startTraining()
console.log(training)

function startTraining() {
    for (let i = 0; i < trainData.length; i++) {
        const {pose, label} = trainData[i];
        nn.addData(pose, {label});
    }
    nn.normalizeData()
    nn.train({
        epochs: 100,
        learningRate: 0.2,
        hiddenUnits: 20,
    }, () => finishedTraining())
}

async function finishedTraining() {
    const results = await nn.classify([0.9668662548065186, 0.69655841588974, -7.258213372551836e-7, 0.9586669206619263, 0.5463995337486267, 0.012671118602156639, 0.8988415002822876, 0.4159107804298401, -0.0005187467322684824, 0.8117558360099792, 0.3555793762207031, -0.01614532619714737, 0.744749128818512, 0.36096033453941345, -0.030586017295718193, 0.8876590728759766, 0.3035179078578949, -0.04591149464249611, 0.7294259071350098, 0.29246020317077637, -0.07461429387331009, 0.7421088814735413, 0.36190271377563477, -0.07818591594696045, 0.7789592742919922, 0.3793448805809021, -0.07760147005319595, 0.8697478771209717, 0.3799988627433777, -0.06805551052093506, 0.7044745683670044, 0.38063475489616394, -0.09175805002450943, 0.7343388795852661, 0.44091352820396423, -0.07747159153223038, 0.7785786986351013, 0.44978663325309753, -0.068380206823349, 0.8451229333877563, 0.4719167947769165, -0.08753705024719238, 0.6991825103759766, 0.4706023633480072, -0.1064990907907486, 0.7356753349304199, 0.5227581262588501, -0.07967950403690338, 0.7807908654212952, 0.5280911922454834, -0.06266561895608902, 0.8187924027442932, 0.5696772336959839, -0.10539513826370239, 0.7105764746665955, 0.5547231435775757, -0.11909715086221695, 0.7414050698280334, 0.5890126824378967, -0.10130401700735092, 0.7802282571792603, 0.595915675163269, -0.08643309772014618
    ])
    console.log(results[0].label);
    // nn.save("model", () => console.log("model was saved!"))
}

const flip = document.getElementById("test");
flip.addEventListener("click", test);

async function test() {
    console.log(training)
    const results = await nn.classify(training)
    console.log(results[0].label);
}

const modelDetails = {
    model: 'model.json',
    metadata: 'model_meta.json',
    weights: 'model.weights.bin'
}
nn.load(modelDetails, () => console.log("het model is geladen!"))

const flop = document.getElementById("accuracy");
flop.addEventListener("click", accuracy);

const confusionMatrixElement = document.getElementById("confusionMatrix");

async function accuracy() {
    let correctPredictions = 0;

    for (let i = 0; i < testData.length; i++) {
        const {pose, label} = testData[i];
        const prediction = await nn.classify(pose);
        const predictedLabel = prediction[0].label;

        console.log(`Predicted: ${predictedLabel}, Actual: ${label}`);

        if (predictedLabel === label) {
            correctPredictions++;
        }
    }

    const accuracy = correctPredictions / testData.length;
    console.log(`Accuracy: ${accuracy * 100}%`);
}

// // Functie om de data willekeurig te sorteren
// data = data.sort(() => Math.random() - 0.5);
//
// // Controleer of de data willekeurig is gesorteerd
// console.log(data);

