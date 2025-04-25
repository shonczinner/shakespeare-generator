let session, vocab, invVocab;

async function setup() {
  // Load the ONNX model using ONNX Runtime Web (ort)
  session = await ort.InferenceSession.create("model.onnx");

  console.log("Model input names:", session.inputNames);
  console.log("Model output names:", session.outputNames);

  
  // Fetch the vocab.json file to get character-to-index mapping
  const res = await fetch("vocab.json");
  vocab = await res.json();
  
  // Reverse the vocab to get index-to-character mapping for decoding
  invVocab = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));
}

function encode(text) {
  // Convert the text into an array of indices based on the vocab
  return text.split("").map(c => vocab[c] ?? vocab[" "]); // Use space as a fallback
}

function decode(indexArray) {
  // Convert the indices back to characters
  return indexArray.map(i => invVocab[i] ?? " ").join("");
}

function sample(probs) {
    // Grab the temperature value from the input field inside the function
    const temperature = parseFloat(document.getElementById("temperature").value) || 1.0;  // Default to 1.0 if no value is provided

    // Apply temperature to logits: temperature > 1 makes distribution flatter, < 1 makes it sharper
    const adjusted = probs.map(p => p / temperature);
  
    // Convert logits to probabilities using softmax
    const maxLogit = Math.max(...adjusted);  // For numerical stability
    const exps = adjusted.map(p => Math.exp(p - maxLogit));  // Subtract maxLogit for better precision
    const sum = exps.reduce((a, b) => a + b, 0);
    const softmax = exps.map(e => e / sum);
  
    // Sample from the distribution
    const rand = Math.random();
    let cumulative = 0;
    for (let i = 0; i < softmax.length; i++) {
        cumulative += softmax[i];
        if (rand < cumulative) return i;
    }
  
    // Fallback in case of rounding errors or other issues
    return softmax.length - 1;
}

  

function getLastTimeStep(outputTensor) {
    const [batchSize, seqLen, vocabSize] = outputTensor.dims;
    const flat = outputTensor.cpuData;
  
    // Offset to start of last timestep for batch 0
    const start = (0 * seqLen + (seqLen - 1)) * vocabSize;
  
    return flat.slice(start, start + vocabSize);  // returns an array of size [65]
  }
  




// Perform inference, generating new text
async function runInference(inputTensor, hiddenTensor) {
    // Set up the feeds object with both input and hidden tensors
    const feeds = {
        input: inputTensor,
        "hidden.1": hiddenTensor
    };

    // Perform inference
    const outputMap = await session.run(feeds);

    // Extract the model's output (logits for next token) and hidden state
    const output = outputMap.output;  // This contains the logits for the next token
    const newHiddenState = outputMap.hidden;  // This contains the updated hidden state

    // Convert logits to probabilities (e.g., using softmax) and sample the next token
    const probs = Array.from(getLastTimeStep(output));  // Convert output to an array of probabilities
    const nextCharIndex = sample(probs);   // Use your sampling method to pick the next token

    // Return the next character and updated hidden state for the next iteration
    return {
        nextChar: invVocab[nextCharIndex],  // Assuming you have invVocab available
        hiddenState: newHiddenState
    };
}

let currentGenId = 0;  // Global generation counter

async function generateText() {
  // Bump the generation ID and capture it locally
  const myGenId = ++currentGenId;

  // Read prompt & numChars
  const prompt = document.getElementById("prompt").value;
  if (!prompt) {
    document.getElementById("output").innerText = "Please enter a prompt.";
    return;
  }
  const numChars = parseInt(document.getElementById("numChars").value) || 0;
  if (numChars < 1) {
    document.getElementById("output").innerText = "Enter a valid number.";
    return;
  }

  // Initialize
  let outputText = prompt;
  let input = encode(prompt);
  let inputTensor = new ort.Tensor('int64', BigInt64Array.from(input.map(BigInt)), [1, input.length]);
  let hiddenTensor = new ort.Tensor('float32', new Float32Array(2 * 1 * 512).fill(0), [2, 1, 512]);

  // Character-by-character loop
  for (let i = 0; i < numChars; i++) {
    // If a new click happened, myGenId != currentGenId → stop old loop
    if (myGenId !== currentGenId) {
      return;
    }

    const result = await runInference(inputTensor, hiddenTensor);
    outputText += result.nextChar;
    hiddenTensor = result.hiddenState;
    inputTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from(encode(result.nextChar).map(BigInt)),
      [1, 1]
    );

    document.getElementById("output").innerText = outputText;

    // small delay so UI can update
    await new Promise(r => setTimeout(r, 0.001));
  }
}


// Initialize everything once the page loads
setup();
