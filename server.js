// server.js
const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors'); // Required for cross-origin requests, useful during development

const app = express();
const port = 3000; // You can change this port if needed

// --- Middleware Configuration ---
app.use(cors()); // Enable CORS for all routes (important for frontend development)
app.use(express.json()); // Enable parsing of JSON request bodies

// Serve static files from the 'public' directory
// Make sure you have a 'public' folder in your project root with index.html, style.css, script.js
app.use(express.static(path.join(__dirname, 'public')));

// --- Environment variables for the Python process ---
// IMPORTANT: Replace placeholders with your actual keys and paths
// Ensure GOOGLE_APPLICATION_CREDENTIALS points to your service account key file (if using Cloud Vision API)
// Ensure GOOGLE_API_KEY is your Gemini API key
const pythonEnv = {
    ...process.env, // Inherit existing environment variables
    'GOOGLE_API_KEY': 'AIzaSyBcj6hnV-aZeQcjYvSGIdARbJIHPohUbwY' 
};

// --- API Route for Chatbot Interaction ---
app.post('/chat', (req, res) => {
    const { query, grade, rag_status } = req.body; // Destructure data from the frontend

    // Basic input validation
    if (!query || !grade || !rag_status) {
        return res.status(400).json({ error: 'Missing parameters: query, grade, or rag_status.' });
    }

    // Path to your main Python script
    const pythonScriptPath = path.join(__dirname, 'integration_llm.py');

    console.log(`Received request: Query="${query}", Grade=${grade}, RAG Status="${rag_status}"`);
    console.log(`Attempting to spawn Python script: ${pythonScriptPath}`);

    // Spawn the Python process with arguments
    // Arguments are passed as an array of strings
    const pythonProcess = spawn('python', [
        pythonScriptPath,
        query,
        grade.toString(), // Ensure grade is passed as a string
        rag_status // 'on' or 'off'
    ], {
        env: pythonEnv, // Pass the defined environment variables to the Python process
        stdio: ['pipe', 'pipe', 'pipe'] // Explicitly pipe stdin, stdout, stderr
    });

    let pythonOutput = ''; // To capture stdout from Python script
    let pythonError = '';  // To capture stderr from Python script

    // Capture stdout data
    pythonProcess.stdout.on('data', (data) => {
        pythonOutput += data.toString();
        // console.log('Python STDOUT chunk:', data.toString()); // Uncomment for debugging Python output
    });

    // Capture stderr data
    pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
        console.error('Python STDERR chunk:', data.toString()); // Log Python errors immediately for debugging
    });

    // Handle process close event
    pythonProcess.on('close', (code) => {
        if (code === 0) {
            // Python script exited successfully
            console.log('Python script exited successfully. Final output:', pythonOutput);
            res.json({ response: pythonOutput.trim() });
        } else {
            // Python script exited with an error code
            console.error(`Python script exited with code ${code}. Full error:`, pythonError);
            res.status(500).json({ error: `Python script error: ${pythonError || 'Unknown error'}` });
        }
    });

    // Handle errors in spawning the process itself (e.g., 'python' command not found)
    pythonProcess.on('error', (err) => {
        console.error('Failed to start python process:', err);
        res.status(500).json({ error: `Server error starting Python script: ${err.message}` });
    });
});

// --- Start the Node.js server ---
app.listen(port, () => {
    console.log(`Node.js server running at http://localhost:${port}`);
    console.log(`Access the demo interface via http://localhost:${port}/`);
    console.log("Ensure 'curriculum_faiss_index_langchain' folder and 'integration_llm.py' are in the same directory as server.js");
});
