document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    // Improved message formatting
    function addMessage(content, isUser = false, context = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
        
        // Add context if available
        if (context) {
            const contextDiv = document.createElement('div');
            contextDiv.className = 'context-info';
            contextDiv.innerHTML = `
                <div class="context-source">
                    <strong>Source:</strong> ${context.metadata.source}
                </div>
                <div class="context-content">
                    ${context.content.replace(/\n/g, '<br>')}
                </div>
            `;
            messageDiv.appendChild(contextDiv);
        }
        
        // Process message content
        const processedContent = content
            .replace(/\n{2,}/g, '<br><br>')  // Multiple newlines
            .replace(/\n/g, '<br>')         // Single newlines
            .replace(/```(\w+)?\n([\s\S]*?)\n?```/g, (match, lang, code) => {
                return `<pre class="code-block"><code class="language-${lang || ''}">${code.replace(/\n/g, '<br>')}</code></pre>`;
            });

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.innerHTML = processedContent;
        
        // Add timestamp for AI messages
        if (!isUser) {
            const timestamp = document.createElement('div');
            timestamp.className = 'message-timestamp';
            timestamp.textContent = new Date().toLocaleTimeString();
            textDiv.prepend(timestamp);
        }

        messageDiv.appendChild(textDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to get context from MCP server
    async function getContext(query) {
        try {
            const response = await fetch('http://localhost:8000/context/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    n_results: 2
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data.contexts;
        } catch (error) {
            console.error('Error getting context:', error);
            return null;
        }
    }

    // Function to send message to Ollama with context
    async function sendToOllama(message, context = null) {
        try {
            // Prepare prompt with context if available
            let prompt = message;
            if (context) {
                prompt = `Context:\n${context.content}\n\nQuestion:\n${message}\n\nAnswer based on the context provided:`;
            }

            const response = await fetch('http://localhost:11434/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'phi4',
                    prompt: prompt,
                    stream: false,
                    options: {
                        temperature: 0.7,
                        top_p: 0.9,
                        num_ctx: 4096,
                        format: 'markdown'
                    }
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
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

        try {
            // Get context from MCP server
            const contexts = await getContext(message);
            let response;

            if (contexts && contexts.length > 0) {
                // Sort contexts by relevance score
                const bestContext = contexts.reduce((prev, current) => 
                    (current.relevance_score > prev.relevance_score) ? current : prev
                );

                // Get AI response with context
                response = await sendToOllama(message, bestContext);
                
                // Add context and response to chat
                addMessage(response, false, bestContext);
            } else {
                // Get AI response without context
                response = await sendToOllama(message);
                addMessage(response, false);
            }
        } catch (error) {
            console.error('Error in chat flow:', error);
            addMessage('Error: Failed to process message. Please try again.', false);
        }

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

    // Initial greeting
    addMessage('Hello! I am Phi-4 with RAG capabilities. How can I help you today?');
});