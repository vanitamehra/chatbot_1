// frontend/js/chatbot.js

// Get HTML elements
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Function to append messages in chat
function appendMessage(sender, message) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('p-2', 'rounded', 'max-w-xs');
    msgDiv.style.wordBreak = "break-word";

    if (sender === 'user') {
        msgDiv.classList.add('bg-blue-100', 'self-end');
    } else {
        msgDiv.classList.add('bg-gray-100');
    }

    msgDiv.textContent = message;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Function to send message to backend
async function sendMessageToBackend(message) {
    try {
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: message })
        });

        const data = await response.json();

        // Show full response in console for debugging
        console.log("Backend response:", data);

        appendMessage('bot', data.answer);
    } catch (err) {
        appendMessage('bot', 'Error connecting to backend.');
        console.error("Backend error:", err);
    }
}

// Handle send button click
sendBtn.addEventListener('click', () => {
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage('user', message);
    userInput.value = '';

    sendMessageToBackend(message);
});

// Handle Enter key
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendBtn.click();
});
