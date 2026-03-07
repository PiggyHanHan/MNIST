const MAX_UPLOAD_BYTES = 1 * 1024 * 1024; // 1m图片限制
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const fileInfoEl = document.getElementById('fileInfo');
const previewImg = document.getElementById('preview');
const previewContainer = document.getElementById('previewContainer');
const resultBox = document.getElementById('result');
const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');
const top2El = document.getElementById('top2');

fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    fileInfoEl.textContent = `${file.name}`;
    previewImage(file);
    uploadBtn.disabled = false;
});

uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    uploadBtn.disabled = true;
    uploadBtn.textContent = '识别中...';
    resultBox.hidden = true;

    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    renderResult(data);

    uploadBtn.disabled = false;
    uploadBtn.textContent = '上传并识别';
});

function renderResult(data) {
    predictionEl.textContent = data.prediction;
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    const formatted = data.top2.map(item => `${item.class}: ${(item.prob * 100).toFixed(2)}%`).join(' | ');
    top2El.textContent = `Top2: ${formatted}`;
    resultBox.hidden = false;
}

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        previewContainer.hidden = false;
    };
    reader.readAsDataURL(file);
}
