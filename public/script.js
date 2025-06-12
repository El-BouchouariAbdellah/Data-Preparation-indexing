// public/script.js
document.addEventListener('DOMContentLoaded', () => {
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    const responseArea = document.getElementById('responseArea');
    const ragToggle = document.getElementById('ragToggle');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorIndicator = document.getElementById('errorIndicator');

    askButton.addEventListener('click', async () => {
        const question = questionInput.value.trim();
        if (!question) {
            // Using a simple alert for demo; for production, use a custom modal or message.
            alert('Veuillez entrer une question.'); 
            return;
        }

        // Reset UI state
        responseArea.innerHTML = '<p class="text-gray-600 italic">Chargement de la réponse...</p>';
        loadingIndicator.classList.remove('hidden');
        errorIndicator.classList.add('hidden');
        askButton.disabled = true; // Disable button to prevent multiple submissions

        try {
            const ragEnabled = ragToggle.checked; // true if ON, false if OFF
            const studentGrade = 6; // Fixed grade for this demo as requested

            // Send request to Node.js backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: question,
                    grade: studentGrade,
                    rag_status: ragEnabled ? 'on' : 'off' // Pass 'on' or 'off' string
                })
            });

            if (!response.ok) {
                // If HTTP status is not 2xx, throw an error
                const errorData = await response.json().catch(() => ({ error: 'Unknown server error' }));
                throw new Error(`HTTP error ${response.status}: ${errorData.error || response.statusText}`);
            }

            const data = await response.json(); // Parse the JSON response from Node.js
            responseArea.innerHTML = `<p class="text-gray-800 whitespace-pre-wrap">${data.response}</p>`; // Use whitespace-pre-wrap to maintain line breaks
        } catch (error) {
            console.error('Error communicating with backend or Python script:', error);
            responseArea.innerHTML = '<p class="text-red-600">Désolé, une erreur est survenue lors de la récupération de la réponse.</p>';
            errorIndicator.classList.remove('hidden');
        } finally {
            // Always hide loading and re-enable button
            loadingIndicator.classList.add('hidden');
            askButton.disabled = false;
        }
    });
});
