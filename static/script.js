const form = document.getElementById("uploadForm");
const loader = document.getElementById("loader");
form.addEventListener("submit", () => {
  loader.classList.remove("hidden");
});
setInterval(() => {
  const frame = document.getElementById("logFrame");
  if (frame) frame.src = "/logs?" + Date.now();
}, 4000);
