// Speech Recognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();

// Speech Synthesis
const synth = window.speechSynthesis;

// States
let isProcessing = false;

// Get the animation container
const animationContainer = document.getElementById('animation-container');

// Function to start listening to user input
const startListening = () => {
  if (isProcessing) return;
  recognition.start();
}

// Function to stop listening to user input
const stopListening = () => {
  recognition.stop();
}

// Function to process the input and produce output
const processInput = (input) => {
  isProcessing = true;
  // Simulate processing time
  setTimeout(() => {
    const output = `You said: ${input}`; // Replace with actual processing logic
    speak(output);
  }, 2000);
}

// Function to speak the output
const speak = (output) => {
  const utterance = new SpeechSynthesisUtterance(output);
  utterance.onend = () => {
    isProcessing = false;
    startListening();
  };
  synth.speak(utterance);
}

// Event when speech recognition returns a result
recognition.addEventListener('result', (e) => {
  const input = e.results[0][0].transcript;
  stopListening();
  processInput(input);
});

// Event when speech recognition ends
recognition.addEventListener('end', () => {
  if (!isProcessing) startListening();
});

// Event when the animation container is clicked
animationContainer.addEventListener('click', () => {
  if (synth.speaking) {
    synth.cancel();
    isProcessing = false;
    startListening();
  }
});

// Start listening initially
startListening();
