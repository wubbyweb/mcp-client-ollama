document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    // Function to add a message to the chat
    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to check if Ollama is accessible
    async function checkOllamaConnection() {
        try {
            const response = await fetch('http://localhost:11434/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'phi4',
                    prompt: 'test',
                    stream: false
                })
            });
            
            if (response.status === 403) {
                throw new Error('CORS_ERROR');
            }
            return true;
        } catch (error) {
            if (error.message === 'CORS_ERROR') {
                addMessage('Error: CORS is not enabled. Please restart Ollama with:\n\n1. First stop Ollama if it\'s running\n2. Then run: OLLAMA_ORIGINS=* ollama serve\n3. Refresh this page after Ollama restarts');
            } else if (error.name === 'TypeError') {
                addMessage('Error: Cannot connect to Ollama. Please ensure Ollama is running on http://localhost:11434');
            }
            return false;
        }
    }

    // Function to send message to Ollama
    async function sendToOllama(message) {
        try {
            const response = await fetch('http://localhost:11434/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'phi4',
                    prompt: message,
                    stream: false,
                    options: {
                        temperature: 0.7,
                        top_p: 0.9
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data.response;
        } catch (error) {
            console.error('Error:', error);
            return 'Error: Failed to get response. Please check the console for details.';
        }
    }

    // Handle send button click
    async function handleSend() {
        const message = userInput.value.trim();
        if (!message) return;

        // Disable input and button while processing
        userInput.disabled = true;
        sendButton.disabled = true;

        // Add user message to chat
        addMessage(message, true);
        userInput.value = '';

        // Get AI response
        const response = await sendToOllama(message);
        addMessage(response);

        // Re-enable input and button
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    }

    // Event listeners
    sendButton.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Check connection and show initial message
    checkOllamaConnection().then(isConnected => {
        if (isConnected) {
            addMessage('Hello! I am Phi-4. How can I help you today?');
        }
    });
});