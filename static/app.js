const $ = id => document.getElementById(id);
let ready = false, selected = null, objectUrl = null, lastResult = null, generation = 0, busy = false;
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function updateButton() { $('analyze').disabled = !ready || !selected || busy; }
function clearResult() { lastResult = null; $('result').hidden = true; $('empty-result').hidden = false; showError(''); }
function choose(file) {
  generation++; selected = null; clearResult();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  $('preview-wrap').hidden = true; $('drop-zone').hidden = false;
  if (!file) { updateButton(); return; }
  if (!['image/jpeg','image/png'].includes(file.type) || file.size > 10 * 1024 * 1024) {
    showError('Choose a JPEG or PNG image smaller than 10 MB.'); updateButton(); return;
  }
  selected = file; objectUrl = URL.createObjectURL(file); $('preview').src = objectUrl;
  $('file-name').textContent = file.name; $('preview-wrap').hidden = false; $('drop-zone').hidden = true;
  updateButton();
}
$('image-input').addEventListener('change', e => choose(e.target.files[0]));
$('clear').addEventListener('click', () => { $('image-input').value = ''; choose(null); });
$('preview').addEventListener('error', () => { choose(null); showError('This image could not be decoded. Choose another JPEG or PNG.'); });
['dragenter','dragover'].forEach(event => $('drop-zone').addEventListener(event, e => { e.preventDefault(); $('drop-zone').classList.add('dragging'); }));
['dragleave','drop'].forEach(event => $('drop-zone').addEventListener(event, e => { e.preventDefault(); $('drop-zone').classList.remove('dragging'); }));
$('drop-zone').addEventListener('drop', e => choose(e.dataTransfer.files[0]));
$('analyze').addEventListener('click', async () => {
  const current = generation; busy = true; updateButton(); clearResult(); $('activity').textContent = 'Analyzing tissue patterns…';
  const form = new FormData(); form.append('image', selected);
  try {
    const response = await fetch('/api/predict', {method:'POST', body:form});
    const result = await response.json();
    if (current !== generation) return;
    if (!response.ok) throw new Error(result.error || 'Prediction failed.');
    lastResult = result; $('prediction').textContent = result.label; $('scores').replaceChildren();
    result.scores.sort((a,b) => b.score-a.score).forEach(item => {
      const row = document.createElement('div'); row.className = 'score-row';
      const head = document.createElement('div'); head.className = 'score-head';
      const label = document.createElement('span'); label.textContent = item.label;
      const score = document.createElement('strong'); score.textContent = (item.score*100).toFixed(1)+'%'; head.append(label,score);
      const bar = document.createElement('div'); bar.className = 'bar'; const fill = document.createElement('span'); fill.style.width = (item.score*100)+'%'; bar.append(fill); row.append(head,bar); $('scores').append(row);
    });
    $('timing').textContent = `Server processing: ${result.processing_ms} ms · includes resizing and inference`;
    $('empty-result').hidden = true; $('result').hidden = false;
  } catch(error) { if (current === generation) showError(error.message || 'Unable to connect. Try again.'); }
  finally { busy = false; updateButton(); $('activity').textContent = ''; }
});
$('download').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult,null,2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href=url; link.download='tissue_prediction.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});
(async () => {
  try {
    const response = await fetch('/api/status'); if (!response.ok) throw new Error('Status unavailable');
    const status = await response.json(); ready = status.ready; $('model-status').textContent = status.message;
    $('status-dot').classList.toggle('ready',ready); updateButton();
    $('scope-warning').hidden = status.evaluation_scope !== 'exploratory image split';
  } catch(error) { $('model-status').textContent='Server unavailable. Refresh to retry.'; }
})();
