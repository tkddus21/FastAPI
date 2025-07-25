<!-- routes/Record.svelte -->
<script>
    let recording = false;
    let mediaRecorder;
    let audioChunks = [];
    let resultText = '';
  
    async function startRecording() {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
  
      mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
      };
  
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/mp3' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.mp3');
  
        const response = await fetch('http://127.0.0.1:8000/transcribe/', {
          method: 'POST',
          body: formData
        });
  
        const result = await response.json();
        resultText = result.text;
        audioChunks = [];
      };
  
      mediaRecorder.start();
      recording = true;
    }
  
    function stopRecording() {
      mediaRecorder.stop();
      recording = false;
    }
  </script>
  
  <h1>🎤 Whisper 음성 인식</h1>
  
  <button on:click={recording ? stopRecording : startRecording}>
    {recording ? '🛑 녹음 중지' : '🎙️ 녹음 시작'}
  </button>
  
  {#if resultText}
    <h2>📝 변환 결과:</h2>
    <p>{resultText}</p>
  {/if}
  