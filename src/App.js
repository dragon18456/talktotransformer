import React, { useState, useRef, useEffect } from 'react';
import TextareaAutosize from 'react-textarea-autosize';
import './App.css';


const imgStyle = {
  width: '100px',
  height: '100px'
}



function App() {
  const instructions = "Type something to tell the transformer or speak it below!"
  const [data, setData] = useState("");
  const [image, setImage] = useState(null);
  const [start, setStart] = useState(false);
  
  let audioElement = document.createElement('audio');
  audioElement.setAttribute('controls', true);

  useEffect(() => {
    fetch('/api/image/?filename=logo').then(res => {
      setImage(res.url)
    });
  }, []);

  async function startDemo() {
    //tts(instructions)
    setStart(true)
    setData(instructions)
  }

  function tts(sentence) {
    fetch('/api/tts?sentence=' + sentence).then(res => {
      audioElement.src = res.url
      audioElement.play()
    });
  }

  async function languageModel(val) {
    const input = document.getElementById("textbox").value
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentence: input })
    };
    setData("loading...")
    const response = await fetch('/api/transformer', requestOptions)
    const dat = await response.json();
    const str = dat.string
    setData(str)

    //tts(str)
  }

  async function asr(val) {
    if (recording.available) {
      let blob = new Blob(recording.blob, { type: 'audio/wav' });
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onload = () => {
        let base64data = reader.result.split(',')[1];
        //console.log(reader.result)
        fetch('/api/asr', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: base64data })
        }).then(res => res.json()).then(data => {
          console.log(data)
          document.getElementById("textbox").value = data.string
        });
      };
    }
  }

  async function clear(val) {
    document.getElementById("textbox").value = ''
    setData(instructions)
  }

  //code reference here: https://codesandbox.io/s/audio-experiments-w7yz8?file=/src/App.js:417-422

  const [stream, setStream] = useState({
    access: false,
    recorder: null,
    error: ""
  });

  const [recording, setRecording] = useState({
    active: false,
    available: false,
    url: "",
    blob: null
  });

  const chunks = useRef([]);

  function getAccess() {
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((mic) => {
        let mediaRecorder;

        try {
          mediaRecorder = new MediaRecorder(mic, {
            mimeType: "audio/webm",
          });
        } catch (err) {
          console.log(err);
        }

        const track = mediaRecorder.stream.getTracks()[0];
        track.onended = () => console.log("ended");

        mediaRecorder.onstart = function () {
          setRecording({
            active: true,
            available: false,
            url: "",
            blob: null
          });
        };

        mediaRecorder.ondataavailable = function (e) {
          console.log("data available");
          chunks.current.push(e.data);
        };

        mediaRecorder.onstop = async function () {
          console.log("stopped");

          const url = URL.createObjectURL(chunks.current[0]);
          const blob = chunks.current
          chunks.current = [];

          setRecording({
            active: false,
            available: true,
            url,
            blob
          });
        };

        setStream({
          ...stream,
          access: true,
          recorder: mediaRecorder
        });
      })
      .catch((error) => {
        console.log(error);
        setStream({ ...stream, error });
      });
  }

  /*const ImageContainer = ({imageUrl}) => {
    return (
      <img src={imageUrl} style = {imgStyle}></img>
    );
  };*/

  return (
    <div className="App">
      <h1>Talk to a Transformer</h1>
      <img src={image} style={imgStyle} ></img>
      <div> {data} </div>
      {start ? (
        <>
          <h2> Your order </h2>
          <p>
            <TextareaAutosize style={{ minWidth: 400}} rows={3} autoFocus type="text" id="textbox" />
          </p>
          <button onClick={languageModel}> Talk to Transformer</button>
          <button onClick={clear}> Clear </button>
        </>
      ) : (
        <button onClick={startDemo}>Start Talking to a Transformer</button>
      )}
      <h2>Record Something to tell the Transformer!</h2>
        {stream.access ? (
        <div className="audio-container">
          <p>
            <button
              className={recording.active ? "active" : null}
              onClick={() => !recording.active && stream.recorder.start()}
            >
              Start Recording
          </button>
            <button onClick={() => recording.active && stream.recorder.stop()}>Stop Recording</button>
            <button id="asr" onClick={asr}>Asr</button>
          </p>
          {recording.available && <audio controls src={recording.url} />}

        </div>
      ) : (
        <button onClick={getAccess}>Get Mic Access</button>
      )}
    </div>
  );
}

export default App;