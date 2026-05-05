const fileEl = document.getElementById("fileInput");
const cameraEl = document.getElementById("cameraInput");
const cameraBtn = document.getElementById("cameraBtn");
const captureBtn = document.getElementById("captureBtn");
const predictBtn = document.getElementById("predictBtn");

const previewImage = document.getElementById("previewImage");
const cameraPreview = document.getElementById("cameraPreview");
const placeholderState = document.getElementById("placeholderState");
const fileName = document.getElementById("fileName");

const resetBtn = document.getElementById("resetBtn");
const resultPopup = document.getElementById("resultPopup");
const closePopupBtn = document.getElementById("closePopupBtn");
const popupFruit = document.getElementById("popupFruit");
const popupResult = document.getElementById("popupResult");

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
    console.error("Image failed to load");
  };
}

async function startCamera() {
  try {
    stopCameraStream();
    fileEl.value = "";

    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    currentStream = stream;
    hasCapturedPhoto = false;

    previewImage.classList.add("hidden");
    placeholderState.classList.add("hidden");

    cameraPreview.srcObject = stream;
    cameraPreview.classList.remove("hidden");

    fileName.textContent = "Camera is on";
  } catch (err) {
    console.error("Could not access camera:", err);
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
  if (cameraEl) {
    cameraEl.value = "";
  }

  previewImage.src = "";
  previewImage.classList.add("hidden");
  placeholderState.classList.remove("hidden");

  fileName.textContent = "No image selected";
  popupFruit.textContent = "-";
  popupResult.textContent = "-";

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

async function dataUrlToFile(dataUrl, filename = "capture.png") {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || "image/png" });
}

async function getImageForPrediction() {
  if (fileEl.files && fileEl.files[0]) {
    return fileEl.files[0];
  }

  if (hasCapturedPhoto && previewImage.src) {
    return await dataUrlToFile(previewImage.src);
  }

  return null;
}

function formatPrediction(data) {
  if (!data || data.fruit === "Unknown") {
    return {
      fruitText: "Unknown",
      detailText: "Not one of the supported fruits or model not confident",
    };
  }

  const freshness =
    data.freshness === "fresh"
      ? "Fresh"
      : data.freshness === "rotten"
        ? "Rotten"
        : "Unknown";

  const confidence =
    typeof data.confidence === "number"
      ? `${Math.round(data.confidence * 100)}% confidence`
      : "Confidence unavailable";

  return {
    fruitText: data.fruit,
    detailText: `${freshness} • ${confidence}`,
  };
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

predictBtn.addEventListener("click", async () => {
  const imageFile = await getImageForPrediction();

  if (!imageFile) {
    alert("Please upload an image or take a picture first.");
    return;
  }

  const originalLabel = predictBtn.textContent;
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  try {
    const form = new FormData();
    form.append("file", imageFile);

    const response = await fetch("/predict", {
      method: "POST",
      body: form,
    });

    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const errData = await response.json();
        if (errData.detail) detail = errData.detail;
      } catch {}
      throw new Error(detail);
    }

    let data = null;
    try {
      data = await response.json();
    } catch {
      throw new Error("Server returned an invalid response.");
    }

    const formatted = formatPrediction(data);
    popupFruit.textContent = formatted.fruitText;
    popupResult.textContent = formatted.detailText;
    openPopup();
  } catch (err) {
    console.error("Prediction error:", err);
    alert(err.message || "Something went wrong during prediction.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = originalLabel;
  }
});
