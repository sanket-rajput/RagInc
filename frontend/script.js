async function sendMessage() {
    const inputField = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const question = inputField.value.trim();

    if (question === "") return;

    // 1. Add User Message to Chat
    addMessage(question, "user-message");
    inputField.value = "";

    // 2. Show Loading Indicator (Optional UI polish)
    const loadingId = "loading-" + Date.now();
    const loadingDiv = document.createElement("div");
    loadingDiv.id = loadingId;
    loadingDiv.className = "message bot-message";
    loadingDiv.innerText = "Thinking...";
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // 3. Call Python Server
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();
        
        // 4. Remove Loading & Show Answer
        document.getElementById(loadingId).remove();
        addMessage(data.answer, "bot-message");

    } catch (error) {
        document.getElementById(loadingId).remove();
        addMessage("Error: Could not connect to server.", "bot-message");
        console.error(error);
    }
}

function addMessage(text, className) {
    const chatBox = document.getElementById("chat-box");
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${className}`;
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function handleEnter(event) {
    if (event.key === "Enter") sendMessage();
}