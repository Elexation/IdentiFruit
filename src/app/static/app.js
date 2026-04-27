const fileEl = document.getElementById("fileInput");
const cameraEl = document.getElementById("cameraInput");
const cameraBtn = document.getElementById("cameraBtn");
const captureBtn = document.getElementById("captureBtn");

const previewImage = document.getElementById("previewImage");
const cameraPreview = document.getElementById("cameraPreview");
const placeholderState = document.getElementById("placeholderState");
const fileName = document.getElementById("fileName");
const resetBtn = document.getElementById("resetBtn");

const resultPopup = document.getElementById("resultPopup");
const closePopupBtn = document.getElementById("closePopupBtn");
const popupFruit = document.getElementById("popupFruit");
const popupRotten = document.getElementById("popupRotten");

let currentObjectUrl = null;
let currentStream = null;
let hasCapturedPhoto = false;

function stopCameraStream() {
  if (currentStream) {
    currentStream.getTracks().forEach((track) => track.stop());
    currentStream = null;
  }

  if (cameraPreview) {
    cameraPreview.srcObject = null;
    cameraPreview.classList.add("hidden");
  }
}

function showPreview(file) {
  if (!file) return;

  stopCameraStream();
  hasCapturedPhoto = false;

  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
  }

  currentObjectUrl = URL.createObjectURL(file);
  previewImage.src = currentObjectUrl;
  fileName.textContent = file.name;

  previewImage.onload = () => {
    previewImage.classList.remove("hidden");
    placeholderState.classList.add("hidden");
  };

  previewImage.onerror = () => {
    console.log("image failed to load");
  };
}

async function startCamera() {
  try {
    stopCameraStream();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    currentStream = stream;
    hasCapturedPhoto = false;

    previewImage.classList.add("hidden");
    placeholderState.classList.add("hidden");

    cameraPreview.srcObject = stream;
    cameraPreview.classList.remove("hidden");

    fileName.textContent = "Camera is on";
  } catch (err) {
    console.log("Could not access camera:", err);
    alert("Camera access was denied or unavailable.");
  }
}

function capturePhoto() {
  if (!currentStream || !cameraPreview.srcObject) {
    alert("Please turn on the camera first.");
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = cameraPreview.videoWidth;
  canvas.height = cameraPreview.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);

  previewImage.src = canvas.toDataURL("image/png");
  previewImage.classList.remove("hidden");

  hasCapturedPhoto = true;
  fileName.textContent = "Photo captured";

  stopCameraStream();
}

function resetUI() {
  stopCameraStream();
  hasCapturedPhoto = false;

  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }

  fileEl.value = "";
  cameraEl.value = "";
  previewImage.src = "";
  previewImage.classList.add("hidden");

  placeholderState.classList.remove("hidden");
  fileName.textContent = "No image selected";

  popupFruit.textContent = "-";
  popupRotten.textContent = "-";
  closePopup();
}

function openPopup() {
  resultPopup.classList.remove("hidden");
  resultPopup.classList.add("flex");
}

function closePopup() {
  resultPopup.classList.add("hidden");
  resultPopup.classList.remove("flex");
}

fileEl.addEventListener("change", () => {
  const file = fileEl.files[0];
  showPreview(file);
});

cameraBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", capturePhoto);

resetBtn.addEventListener("click", resetUI);
closePopupBtn.addEventListener("click", closePopup);

resultPopup.addEventListener("click", (e) => {
  if (e.target === resultPopup) {
    closePopup();
  }
});

document.getElementById("predictBtn").onclick = () => {
  const uploadedFile = fileEl.files[0];
  const hasUploadedImage = !!uploadedFile;
  const hasWebcamPhoto = hasCapturedPhoto;

  if (!hasUploadedImage && !hasWebcamPhoto) {
    alert("Please upload an image or take a picture first.");
    return;
  }

  const fruits = ["Apple", "Banana", "Strawberry"];
  const fruit = fruits[Math.floor(Math.random() * fruits.length)];
  const rottenPercent = Math.floor(Math.random() * 100);

  popupFruit.textContent = fruit;
  popupRotten.textContent = `${rottenPercent}%`;

  openPopup();
};
