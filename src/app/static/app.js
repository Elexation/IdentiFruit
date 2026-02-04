const fileEl = document.getElementById("file");
const out = document.getElementById("out");

document.getElementById("btn").onclick = async () => {
  // Ensure a file is selected
  if (!fileEl.files.length) return;

  const form = new FormData();
  form.append("file", fileEl.files[0]);

  const res = await fetch("/predict", { method: "POST", body: form });
  const data = await res.json();

  // Display the prediction result
  out.textContent = JSON.stringify(data, null, 2);
};
